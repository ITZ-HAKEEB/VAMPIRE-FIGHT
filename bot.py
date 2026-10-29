import asyncio
import logging
import numpy as np

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from pyrogram import Client, filters, idle
from pyrogram.types import Message

from pytgcalls import PyTgCalls
from pytgcalls import filters as call_filters
from pytgcalls.types import (
    Device,
    Direction,
    ExternalMedia,
    MediaStream,
    RecordStream,
    StreamFrames,
)
from pytgcalls.types.raw import AudioParameters

from config import API_ID, API_HASH, SESSION_STRING, OWNER_ID


# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("TelegramFightVC")


# ================= SUDO USERS =================
SUDO_USERS = {OWNER_ID}


# ================= CLIENT =================
app = Client(
    "tg_fight_userbot",
    api_id=API_ID,
    api_hash=API_HASH,
    session_string=SESSION_STRING,
)
call_py = PyTgCalls(app)


# ================= BRIDGE STATE =================
current_logger_chat = None
current_target_chat = None
USERBOT_ID = None

current_volume = 200       # 1-1000
current_boost = 100        # 1-5000
current_high_volume = 0    # 0-2000
current_covolume = 0       # 0-800
current_echo = 0           # 0-10000
current_gain = 0           # 0-500 (distortion)
current_bass = 0           # 0-300
current_voice = "normal"   # normal, moti, patli

SAMPLE_RATE = 48000
CHANNELS = 2

ECHO_DELAY_SAMPLES = int(SAMPLE_RATE * 0.18) * CHANNELS
echo_buffer = np.zeros(ECHO_DELAY_SAMPLES, dtype=np.float32)
bass_filter_state = np.zeros(CHANNELS, dtype=np.float32)
voice_filter_state = np.zeros(CHANNELS, dtype=np.float32)

bridge_lock = asyncio.Lock()

AUDIO_PARAMETERS = AudioParameters(
    bitrate=48000,
    channels=2,
)


# ================= BOOST CURVES =================
def get_boost_multiplier(value: int) -> float:
    """Map boost 1-5000 to a controlled gain multiplier."""
    value = max(1, min(5000, int(value)))
    if value <= 100:
        return 0.1 + (0.9 * (value - 1) / 99.0)
    return 1.0 + (1.5 * (value - 100) / 4900.0)


def get_high_volume_multiplier(value: int) -> float:
    """Map high-volume 0-2000 to additional gain up to 2x."""
    value = max(0, min(2000, int(value)))
    return 1.0 if value == 0 else 1.0 + value / 2000.0


def get_covolume_multiplier(value: int) -> float:
    """Map co-volume 0-800 to additional gain up to 2.5x."""
    value = max(0, min(800, int(value)))
    return 1.0 if value == 0 else 1.0 + (1.5 * value / 800.0)


# ================= AUTHORIZATION =================
def is_authorized(_, __, message: Message):
    if not message.from_user:
        return False
    return (
        message.from_user.id in SUDO_USERS
        or (
            USERBOT_ID is not None
            and message.from_user.id == USERBOT_ID
        )
    )


sudo_filter = filters.create(is_authorized)


# ================= AUDIO DSP =================
def process_audio(raw_audio: bytes) -> bytes:
    """
    Process stereo signed 16-bit PCM.

    Voice modes use simple frequency shaping:
    - moti: warmer/heavier tone by adding low-frequency content.
    - patli: brighter tone by adding high-frequency content.
    - normal: no voice tone shaping.

    This is EQ/tone shaping, not true pitch shifting.
    """
    global echo_buffer, bass_filter_state, voice_filter_state

    if not raw_audio:
        return raw_audio

    audio_i16 = np.frombuffer(raw_audio, dtype=np.int16)
    if audio_i16.size < CHANNELS:
        return raw_audio

    usable_size = (audio_i16.size // CHANNELS) * CHANNELS
    audio = audio_i16[:usable_size].astype(np.float32).reshape(-1, CHANNELS)

    # 1. Main volume and boost.
    audio *= current_volume / 100.0
    audio *= get_boost_multiplier(current_boost)
    audio *= get_high_volume_multiplier(current_high_volume)
    audio *= get_covolume_multiplier(current_covolume)

    # 2. One-pole low-pass state used for bass and voice shaping.
    # Calculate a smooth low-frequency component independently per channel.
    low = np.empty_like(audio)
    state = voice_filter_state.copy()
    alpha = 0.035
    for i in range(len(audio)):
        state += alpha * (audio[i] - state)
        low[i] = state
    voice_filter_state = state.copy()

    # 3. Voice tone shaping.
    # These modes change tonal balance; they do not perform true pitch shifting.
    if current_voice == "moti":
        audio += low * 0.65
    elif current_voice == "patli":
        high = audio - low
        audio += high * 0.45

    # 4. User bass boost.
    if current_bass > 0:
        bass_amount = current_bass / 100.0
        audio += low * bass_amount

    # 5. Echo.
    if current_echo == 0:
        echo_buffer.fill(0)
    else:
        flat_audio = audio.reshape(-1)
        if len(echo_buffer) != ECHO_DELAY_SAMPLES:
            echo_buffer = np.zeros(ECHO_DELAY_SAMPLES, dtype=np.float32)

        combined = np.concatenate((echo_buffer, flat_audio))
        delayed = combined[:len(flat_audio)]
        echo_buffer = combined[-ECHO_DELAY_SAMPLES:].copy()

        # The 0-10000 setting scales echo strength from 0 to 0.55.
        echo_strength = (current_echo / 10000.0) * 0.55
        flat_audio = flat_audio + delayed * echo_strength
        audio = flat_audio.reshape(-1, CHANNELS)

    # 6. Optional soft distortion.
    if current_gain > 0:
        drive = 1.0 + min(current_gain, 500) / 50.0
        audio = 32767.0 * np.tanh((audio / 32767.0) * drive)

    # 7. Peak limiter: avoids hard clipping, but may reduce perceived loudness.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    target_peak = 30000.0
    if peak > target_peak:
        audio *= target_peak / peak

    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio.reshape(-1).tobytes()


# ================= AUDIO BRIDGE =================
@call_py.on_update(
    call_filters.stream_frame(Direction.INCOMING, Device.MICROPHONE)
)
async def relay_audio(_, update: StreamFrames):
    logger_chat = current_logger_chat
    target_chat = current_target_chat

    if logger_chat is None or target_chat is None:
        return
    if update.chat_id != logger_chat or not update.frames:
        return

    try:
        frame_arrays = [
            np.frombuffer(frame.frame, dtype=np.int16)
            for frame in update.frames
            if frame.frame
        ]
        if not frame_arrays:
            return

        audio = np.concatenate(frame_arrays)
        processed_audio = process_audio(audio.tobytes())
        await call_py.send_frame(
            target_chat,
            Device.MICROPHONE,
            processed_audio,
        )
    except Exception:
        logger.exception("Audio forwarding failed")


# ================= .HELP =================
@app.on_message(filters.command("help", prefixes=".") & sudo_filter)
async def help_command(client: Client, message: Message):
    await message.reply(
        "🔥 **Telegram Fight Voice Chat Bot** 🔥\n\n"
        "• `.help` - Show commands.\n"
        "• `.joinvc <logger_id> <target_id>` - Start bridge.\n"
        "• `.leavevc` - Stop bridge.\n"
        "• `.volume <1-1000>` - Set output volume.\n"
        "• `.boost <1-5000>` - Adjust controlled boost.\n"
        "• `.highvolume <0-2000>` - Set additional volume gain.\n"
        "• `.covolume <0-800>` - Set co-volume gain.\n"
        "• `.echo <0-10000>` - Set echo strength; 0 disables it.\n"
        "• `.voice <moti/patli/normal>` - Change voice tone.\n"
        "• `.gain <0-500>` - Set distortion; 0 disables it.\n"
        "• `.bass <0-300>` - Set bass boost; 0 disables it.\n"
        "• `.addsudo <user_id>` - Add sudo user.\n"
        "• `.rmsudo <user_id>` - Remove sudo user.\n"
        "• `.sudolist` - Show sudo users."
    )


# ================= .ADDSUDO =================
@app.on_message(filters.command("addsudo", prefixes=".") & filters.user(OWNER_ID))
async def add_sudo(client: Client, message: Message):
    try:
        args = (message.text or "").split()
        if len(args) < 2:
            await message.reply("Usage: `.addsudo <user_id>`")
            return

        user_id = int(args[1])
        if user_id <= 0:
            await message.reply("❌ Enter a valid positive user ID.")
            return

        SUDO_USERS.add(user_id)
        await message.reply(f"✅ Added `{user_id}` to sudo users.")
    except ValueError:
        await message.reply("❌ Enter a valid numeric user ID.")
    except Exception as e:
        logger.exception("Add sudo failed")
        await message.reply(f"❌ Error: {e}")


# ================= .RMSUDO =================
@app.on_message(filters.command("rmsudo", prefixes=".") & filters.user(OWNER_ID))
async def remove_sudo(client: Client, message: Message):
    try:
        args = (message.text or "").split()
        if len(args) < 2:
            await message.reply("Usage: `.rmsudo <user_id>`")
            return

        user_id = int(args[1])
        if user_id == OWNER_ID:
            await message.reply("❌ You cannot remove the owner.")
            return
        if user_id not in SUDO_USERS:
            await message.reply("⚠️ User is not in the sudo list.")
            return

        SUDO_USERS.remove(user_id)
        await message.reply(f"✅ Removed `{user_id}` from sudo users.")
    except ValueError:
        await message.reply("❌ Enter a valid numeric user ID.")
    except Exception as e:
        logger.exception("Remove sudo failed")
        await message.reply(f"❌ Error: {e}")


# ================= .SUDOLIST =================
@app.on_message(filters.command("sudolist", prefixes=".") & sudo_filter)
async def sudo_list(client: Client, message: Message):
    users = "\n".join(f"• `{user_id}`" for user_id in sorted(SUDO_USERS))
    await message.reply("👑 **Authorized Sudo Users**\n\n" + (users or "No sudo users."))


# ================= .JOINVC =================
@app.on_message(filters.command("joinvc", prefixes=".") & sudo_filter)
async def join_vc(client: Client, message: Message):
    global current_logger_chat, current_target_chat
    global echo_buffer, bass_filter_state, voice_filter_state

    args = (message.text or "").split()
    if len(args) < 3:
        await message.reply("Usage: `.joinvc <logger_group_id> <target_group_id>`")
        return

    try:
        logger_chat = int(args[1])
        target_chat = int(args[2])
        if logger_chat == target_chat:
            await message.reply("❌ Logger and target must be different groups.")
            return
    except ValueError:
        await message.reply("❌ Both group IDs must be numeric.")
        return

    async with bridge_lock:
        if current_logger_chat is not None:
            await message.reply("⚠️ Bridge is already active. Use `.leavevc` first.")
            return

        await message.reply("🔄 Connecting to logger and target voice chats...")
        try:
            for chat_id in (logger_chat, target_chat):
                await call_py.play(
                    chat_id,
                    MediaStream(ExternalMedia.AUDIO, AUDIO_PARAMETERS),
                )
            for chat_id in (logger_chat, target_chat):
                await call_py.record(
                    chat_id,
                    RecordStream(True, AUDIO_PARAMETERS),
                )

            current_logger_chat = logger_chat
            current_target_chat = target_chat
            echo_buffer = np.zeros(ECHO_DELAY_SAMPLES, dtype=np.float32)
            bass_filter_state = np.zeros(CHANNELS, dtype=np.float32)
            voice_filter_state = np.zeros(CHANNELS, dtype=np.float32)

            await message.reply(
                "✅ **Voice bridge initialized**\n\n"
                f"📡 Logger: `{logger_chat}`\n"
                f"🔊 Target: `{target_chat}`\n"
                f"🎚 Volume: `{current_volume}/1000`\n"
                f"📢 Boost: `{current_boost}/5000`\n"
                f"🔊 High Volume: `{current_high_volume}/2000`\n"
                f"🔊 Co-volume: `{current_covolume}/800`\n"
                f"🔁 Echo: `{current_echo}/10000`\n"
                f"🎙 Voice: `{current_voice}`\n"
                f"🎛 Gain: `{current_gain}/500`\n"
                f"🔊 Bass: `{current_bass}/300`\n\n"
                "Audio from the logger group will be processed and forwarded to the target group."
            )
        except Exception as e:
            logger.exception("Failed to initialize voice bridge")
            for chat_id in (logger_chat, target_chat):
                try:
                    await call_py.leave_call(chat_id)
                except Exception:
                    logger.warning("No active call to clean up for group %s", chat_id)

            current_logger_chat = None
            current_target_chat = None
            await message.reply(
                "❌ **Bridge initialization failed**\n\n"
                f"Error: `{type(e).__name__}: {e}`\n\n"
                "Check group membership, group IDs, and voice chat status."
            )


# ================= NUMERIC COMMAND HELPERS =================
async def update_integer_setting(
    message: Message,
    args,
    *,
    name: str,
    minimum: int,
    maximum: int,
    setter,
    current_value: int,
    usage: str,
    extra: str = "",
):
    if len(args) < 2:
        await message.reply(
            f"{name}: `{current_value}/{maximum}`\nUsage: `{usage}`"
            + (f"\n{extra}" if extra else "")
        )
        return None
    try:
        value = int(args[1])
    except ValueError:
        await message.reply("❌ Enter a valid whole number.")
        return None
    if not minimum <= value <= maximum:
        await message.reply(f"⚠️ Enter a value between {minimum} and {maximum}.")
        return None
    setter(value)
    return value


# ================= .VOLUME =================
@app.on_message(filters.command("volume", prefixes=".") & sudo_filter)
async def set_volume(client: Client, message: Message):
    global current_volume
    args = (message.text or "").split()
    value = await update_integer_setting(
        message, args, name="🔊 Current volume", minimum=1, maximum=1000,
        setter=lambda v: None, current_value=current_volume,
        usage=".volume <1-1000>",
    )
    if value is None:
        return
    current_volume = value
    await message.reply(
        f"🔊 Volume setting updated to `{current_volume}/1000`.\n"
        "⚠️ Very high values may distort audio or trigger limiting."
    )


# ================= .BOOST =================
@app.on_message(filters.command("boost", prefixes=".") & sudo_filter)
async def set_boost(client: Client, message: Message):
    global current_boost
    args = (message.text or "").split()
    if len(args) < 2:
        await message.reply(
            f"📢 Current boost: `{current_boost}/5000`\n"
            f"🎚 Effective multiplier: `{get_boost_multiplier(current_boost):.2f}x`\n"
            "Usage: `.boost <1-5000>`; use `.boost 100` for neutral boost."
        )
        return
    try:
        value = int(args[1])
        if not 1 <= value <= 5000:
            await message.reply("⚠️ Boost must be between 1 and 5000.")
            return
        current_boost = value
        await message.reply(
            "✅ **Boost updated**\n"
            f"📢 Setting: `{value}/5000`\n"
            f"🎚 Effective multiplier: `{get_boost_multiplier(value):.2f}x`\n"
            "ℹ️ Very high output levels may distort the voice."
        )
    except ValueError:
        await message.reply("❌ Enter a valid whole number.")
    except Exception as e:
        logger.exception("Boost update failed")
        await message.reply(f"❌ Error: {e}")


# ================= .HIGHVOLUME =================
@app.on_message(filters.command("highvolume", prefixes=".") & sudo_filter)
async def set_high_volume(client: Client, message: Message):
    global current_high_volume
    args = (message.text or "").split()
    if len(args) < 2:
        await message.reply(
            f"🔊 Current high volume: `{current_high_volume}/2000`\n"
            f"🎚 Additional multiplier: `{get_high_volume_multiplier(current_high_volume):.2f}x`\n"
            "Usage: `.highvolume <0-2000>`; `.highvolume 0` disables extra gain."
        )
        return
    try:
        value = int(args[1])
        if not 0 <= value <= 2000:
            await message.reply("⚠️ Enter a value between 0 and 2000.")
            return
        current_high_volume = value
        await message.reply(
            "✅ **High Volume Updated**\n"
            f"🔊 Setting: `{value}/2000`\n"
            f"🎚 Additional multiplier: `{get_high_volume_multiplier(value):.2f}x`\n"
            "ℹ️ High settings can distort audio or trigger limiting."
        )
    except ValueError:
        await message.reply("❌ Enter a valid whole number.")
    except Exception as e:
        logger.exception("High volume update failed")
        await message.reply(f"❌ Error: {e}")


# ================= .COVOLUME =================
@app.on_message(filters.command("covolume", prefixes=".") & sudo_filter)
async def set_covolume(client: Client, message: Message):
    global current_covolume
    args = (message.text or "").split()
    if len(args) < 2:
        await message.reply(
            f"🔊 Current co-volume: `{current_covolume}/800`\n"
            f"🎚 Additional multiplier: `{get_covolume_multiplier(current_covolume):.2f}x`\n"
            "Usage: `.covolume <0-800>`"
        )
        return
    try:
        value = int(args[1])
        if not 0 <= value <= 800:
            await message.reply("⚠️ Enter a value between 0 and 800.")
            return
        current_covolume = value
        await message.reply(
            "✅ **Co-volume updated**\n"
            f"🔊 Setting: `{value}/800`\n"
            f"🎚 Additional multiplier: `{get_covolume_multiplier(value):.2f}x`\n"
            "ℹ️ The peak limiter may reduce the effect at very high levels."
        )
    except ValueError:
        await message.reply("❌ Enter a valid whole number.")
    except Exception as e:
        logger.exception("Co-volume update failed")
        await message.reply(f"❌ Error: {e}")


# ================= .ECHO =================
@app.on_message(filters.command("echo", prefixes=".") & sudo_filter)
async def set_echo(client: Client, message: Message):
    global current_echo, echo_buffer
    args = (message.text or "").split()
    if len(args) < 2:
        await message.reply(
            f"🔁 Current echo: `{current_echo}/10000`\n"
            "Usage: `.echo <0-10000>`; `.echo 0` turns echo off."
        )
        return
    try:
        value = int(args[1])
        if not 0 <= value <= 10000:
            await message.reply("⚠️ Echo must be between 0 and 10000.")
            return
        current_echo = value
        if value == 0:
            echo_buffer.fill(0)
            await message.reply("🔁 Echo disabled.")
        else:
            await message.reply(
                f"🔁 Echo set to `{value}/10000` "
                f"(strength {(value / 10000.0) * 0.55:.2f})."
            )
    except ValueError:
        await message.reply("❌ Enter a valid whole number.")
    except Exception as e:
        logger.exception("Echo update failed")
        await message.reply(f"❌ Error: {e}")


# ================= .VOICE =================
@app.on_message(filters.command("voice", prefixes=".") & sudo_filter)
async def set_voice(client: Client, message: Message):
    global current_voice, voice_filter_state
    args = (message.text or "").split()
    if len(args) < 2:
        await message.reply(
            f"🎙 Current voice mode: `{current_voice}`\n\n"
            "Usage:\n"
            "• `.voice moti` - Heavier/warmer tone.\n"
            "• `.voice patli` - Brighter/lighter tone.\n"
            "• `.voice normal` - Normal tone."
        )
        return

    mode = args[1].lower()
    if mode not in ("moti", "patli", "normal"):
        await message.reply("❌ Choose `moti`, `patli`, or `normal`.")
        return

    current_voice = mode
    voice_filter_state.fill(0)
    await message.reply(
        f"✅ Voice mode set to `{mode}`.\n"
        "ℹ️ This is EQ/tone shaping, not a true pitch-shifting effect."
    )


# ================= .GAIN =================
@app.on_message(filters.command("gain", prefixes=".") & sudo_filter)
async def set_gain(client: Client, message: Message):
    global current_gain
    args = (message.text or "").split()
    if len(args) < 2:
        await message.reply(
            f"🎛 Current distortion: `{current_gain}/500`\n"
            "Usage: `.gain <0-500>`; `.gain 0` disables distortion."
        )
        return
    try:
        value = int(args[1])
        if not 0 <= value <= 500:
            await message.reply("⚠️ Gain must be between 0 and 500.")
            return
        current_gain = value
        await message.reply(
            "🎛 Added distortion disabled." if value == 0
            else f"🎛 Distortion set to `{value}/500`."
        )
    except ValueError:
        await message.reply("❌ Enter a valid number.")
    except Exception as e:
        logger.exception("Gain update failed")
        await message.reply(f"❌ Error: {e}")


# ================= .BASS =================
@app.on_message(filters.command("bass", prefixes=".") & sudo_filter)
async def set_bass(client: Client, message: Message):
    global current_bass, bass_filter_state
    args = (message.text or "").split()
    if len(args) < 2:
        await message.reply(
            f"🔊 Current bass: `{current_bass}/300`\n"
            "Usage: `.bass <0-300>`; `.bass 0` disables bass boost."
        )
        return
    try:
        value = int(args[1])
        if not 0 <= value <= 300:
            await message.reply("⚠️ Bass must be between 0 and 300.")
            return
        current_bass = value
        bass_filter_state.fill(0)
        await message.reply(
            "🔊 Bass boost disabled." if value == 0
            else f"🔊 Bass boost set to `{value}/300`."
        )
    except ValueError:
        await message.reply("❌ Enter a valid number.")
    except Exception as e:
        logger.exception("Bass update failed")
        await message.reply(f"❌ Error: {e}")


# ================= .LEAVEVC =================
@app.on_message(filters.command("leavevc", prefixes=".") & sudo_filter)
async def leave_vc(client: Client, message: Message):
    global current_logger_chat, current_target_chat
    global echo_buffer, bass_filter_state, voice_filter_state

    async with bridge_lock:
        if current_logger_chat is None and current_target_chat is None:
            await message.reply("⚠️ No active bridge.")
            return

        chat_ids = {
            chat_id for chat_id in (current_logger_chat, current_target_chat)
            if chat_id is not None
        }
        errors = []
        for chat_id in chat_ids:
            try:
                await call_py.leave_call(chat_id)
            except Exception as e:
                errors.append(f"{chat_id}: {e}")
                logger.warning("Could not leave group %s: %s", chat_id, e)

        current_logger_chat = None
        current_target_chat = None
        echo_buffer.fill(0)
        bass_filter_state.fill(0)
        voice_filter_state.fill(0)

        if errors:
            await message.reply(
                "⚠️ Some voice chats may not have disconnected:\n"
                + "\n".join(errors)
            )
        else:
            await message.reply("🔌 Left both voice chats. Audio bridge stopped.")


# ================= MAIN =================
async def main():
    global USERBOT_ID
    logger.info("Starting Telegram Fight Voice Chat Bot...")

    await app.start()
    try:
        await call_py.start()
        me = await app.get_me()
        USERBOT_ID = me.id
        logger.info("Userbot started as @%s (ID: %s)", me.username, me.id)
        await idle()
    finally:
        try:
            await call_py.stop()
        except Exception:
            logger.exception("Error stopping PyTgCalls")
        try:
            if app.is_connected:
                await app.stop()
        except Exception:
            logger.exception("Error stopping Pyrogram")


# ================= ENTRY POINT =================
if __name__ == "__main__":
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")
    except Exception:
        logger.exception("Fatal startup error.")
    finally:
        if not loop.is_closed():
            loop.close()
