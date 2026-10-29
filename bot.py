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
logging.basicConfig(level=logging.INFO)
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

current_volume = 200
current_boost = 100
current_high_volume = 0
current_covolume = 0
current_echo = 0
current_gain = 0
current_bass = 0
current_voice = "normal"

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


# ================= BOOST CURVE =================
def get_boost_multiplier(value: int) -> float:
    value = max(1, min(5000, int(value)))

    if value <= 100:
        return 0.1 + (0.9 * (value - 1) / 99.0)

    return 1.0 + (1.5 * (value - 100) / 4900.0)


# ================= HIGH VOLUME =================
def get_high_volume_multiplier(value: int) -> float:
    value = max(0, min(2000, int(value)))
    return 1.0 if value == 0 else 1.0 + value / 2000.0


# ================= CO VOLUME =================
def get_covolume_multiplier(value: int) -> float:
    value = max(0, min(800, int(value)))
    return 1.0 + (value / 100.0)


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
    global echo_buffer
    global bass_filter_state
    global voice_filter_state

    if not raw_audio:
        return raw_audio

    audio = np.frombuffer(raw_audio, dtype=np.int16)

    if audio.size < CHANNELS:
        return raw_audio

    usable_size = (audio.size // CHANNELS) * CHANNELS
    audio = audio[:usable_size].astype(np.float32)
    audio = audio.reshape(-1, CHANNELS)

    # 1. Main volume
    audio *= current_volume / 100.0

    # 2. Existing boost
    audio *= get_boost_multiplier(current_boost)

    # 3. High volume
    audio *= get_high_volume_multiplier(current_high_volume)

    # 4. Co-volume
    audio *= get_covolume_multiplier(current_covolume)

    # 5. Voice tone EQ
    # "moti" emphasizes lower frequencies.
    # "patli" emphasizes higher frequencies.
    if current_voice != "normal" and len(audio):
        alpha = 0.055
        low = np.empty_like(audio)
        state = voice_filter_state.copy()

        for i in range(len(audio)):
            state += alpha * (audio[i] - state)
            low[i] = state

        voice_filter_state = state.copy()

        if current_voice == "moti":
            audio = audio + low * 0.65
        elif current_voice == "patli":
            high = audio - low
            audio = audio + high * 0.65

    # 6. Bass enhancement
    if current_bass > 0 and len(audio):
        alpha = 0.03
        low = np.empty_like(audio)
        state = bass_filter_state.copy()

        for i in range(len(audio)):
            state += alpha * (audio[i] - state)
            low[i] = state

        bass_filter_state = state.copy()
        audio += low * (current_bass / 100.0)

    # 7. Echo up to 10000
    if current_echo == 0:
        echo_buffer.fill(0)
    else:
        flat_audio = audio.reshape(-1)

        if len(echo_buffer) != ECHO_DELAY_SAMPLES:
            echo_buffer = np.zeros(
                ECHO_DELAY_SAMPLES,
                dtype=np.float32,
            )

        combined = np.concatenate((echo_buffer, flat_audio))
        delayed = combined[:len(flat_audio)]
        echo_buffer = combined[-ECHO_DELAY_SAMPLES:].copy()

        # Keep echo feedback bounded to avoid runaway audio.
        strength = min(current_echo / 10000.0, 1.0)
        flat_audio = flat_audio + delayed * (0.45 * strength)
        audio = flat_audio.reshape(-1, CHANNELS)

    # 8. Gain / saturation effect up to 7000
    if current_gain > 0:
        # Gradually increase saturation while limiting runaway peaks.
        drive = 1.0 + (min(current_gain, 7000) / 100.0)
        audio = 32767.0 * np.tanh(
            (audio / 32767.0) * drive
        )

    # 9. Soft limiter
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if peak > 30000.0:
        audio *= 30000.0 / peak

    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio.reshape(-1).tobytes()


# ================= AUDIO BRIDGE =================
@call_py.on_update(
    call_filters.stream_frame(
        Direction.INCOMING,
        Device.MICROPHONE,
    )
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


# ================= HELP =================
@app.on_message(filters.command("help", prefixes=".") & sudo_filter)
async def help_command(client: Client, message: Message):
    await message.reply(
        "🔥 **Telegram Fight Voice Chat Bot** 🔥\n\n"
        "• `.help` - Show commands.\n"
        "• `.joinvc <logger_id> <target_id>` - Start bridge.\n"
        "• `.leavevc` - Stop bridge.\n"
        "• `.volume <1-1000>` - Main volume.\n"
        "• `.boost <1-5000>` - Boost control.\n"
        "• `.highvolume <0-2000>` - Extra gain.\n"
        "• `.covolume <0-800>` - Additional output gain.\n"
        "• `.echo <0-10000>` - Echo effect.\n"
        "• `.gain <0-7000>` - Distortion/saturation.\n"
        "• `.bass <0-300>` - Bass enhancement.\n"
        "• `.voice moti` - Deeper/warmer tone.\n"
        "• `.voice patli` - Brighter tone.\n"
        "• `.voice normal` - Normal tone.\n"
        "• `.addsudo <user_id>` - Add sudo.\n"
        "• `.rmsudo <user_id>` - Remove sudo.\n"
        "• `.sudolist` - List sudo users."
    )


# ================= ADDSUDO =================
@app.on_message(
    filters.command("addsudo", prefixes=".") & filters.user(OWNER_ID)
)
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


# ================= RMSUDO =================
@app.on_message(
    filters.command("rmsudo", prefixes=".") & filters.user(OWNER_ID)
)
async def remove_sudo(client: Client, message: Message):
    try:
        args = (message.text or "").split()

        if len(args) < 2:
            await message.reply("Usage: `.rmsudo <user_id>`")
            return

        user_id = int(args[1])

        if user_id == OWNER_ID:
            await message.reply("❌ You cannot remove the owner.")

        elif user_id not in SUDO_USERS:
            await message.reply("⚠️ User is not in the sudo list.")

        else:
            SUDO_USERS.remove(user_id)
            await message.reply(f"✅ Removed `{user_id}` from sudo users.")

    except ValueError:
        await message.reply("❌ Enter a valid numeric user ID.")
    except Exception as e:
        logger.exception("Remove sudo failed")
        await message.reply(f"❌ Error: {e}")


# ================= SUDOLIST =================
@app.on_message(filters.command("sudolist", prefixes=".") & sudo_filter)
async def sudo_list(client: Client, message: Message):
    users = "\n".join(f"• `{uid}`" for uid in sorted(SUDO_USERS))
    await message.reply(
        "👑 **Authorized Sudo Users**\n\n" + (users or "No sudo users.")
    )


# ================= JOINVC =================
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
                f"📣 Co-volume: `{current_covolume}/800`\n"
                f"🔁 Echo: `{current_echo}/10000`\n"
                f"🎛 Gain: `{current_gain}/7000`\n"
                f"🔊 Bass: `{current_bass}/300`\n"
                f"🎙 Voice: `{current_voice}`"
            )

        except Exception as e:
            logger.exception("Failed to initialize voice bridge")

            for chat_id in (logger_chat, target_chat):
                try:
                    await call_py.leave_call(chat_id)
                except Exception:
                    logger.warning("Cleanup failed for group %s", chat_id)

            current_logger_chat = None
            current_target_chat = None

            await message.reply(
                "❌ **Bridge initialization failed**\n"
                f"`{type(e).__name__}: {e}`"
            )


# ================= VOLUME =================
@app.on_message(filters.command("volume", prefixes=".") & sudo_filter)
async def set_volume(client: Client, message: Message):
    global current_volume
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(f"🔊 Current volume: `{current_volume}/1000`")
        return

    try:
        value = int(args[1])
        if not 1 <= value <= 1000:
            await message.reply("⚠️ Enter a value between 1 and 1000.")
            return

        current_volume = value
        await message.reply(f"✅ Volume set to `{value}/1000`.")

    except ValueError:
        await message.reply("❌ Enter a valid whole number.")


# ================= BOOST =================
@app.on_message(filters.command("boost", prefixes=".") & sudo_filter)
async def set_boost(client: Client, message: Message):
    global current_boost
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(
            f"📢 Boost: `{current_boost}/5000`\n"
            f"Multiplier: `{get_boost_multiplier(current_boost):.2f}x`"
        )
        return

    try:
        value = int(args[1])
        if not 1 <= value <= 5000:
            await message.reply("⚠️ Boost must be between 1 and 5000.")
            return

        current_boost = value
        await message.reply(
            f"✅ Boost set to `{value}/5000`\n"
            f"Multiplier: `{get_boost_multiplier(value):.2f}x`"
        )

    except ValueError:
        await message.reply("❌ Enter a valid whole number.")


# ================= HIGHVOLUME =================
@app.on_message(filters.command("highvolume", prefixes=".") & sudo_filter)
async def set_high_volume(client: Client, message: Message):
    global current_high_volume
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(
            f"🔊 High volume: `{current_high_volume}/2000`\n"
            f"Multiplier: `{get_high_volume_multiplier(current_high_volume):.2f}x`"
        )
        return

    try:
        value = int(args[1])
        if not 0 <= value <= 2000:
            await message.reply("⚠️ Enter a value between 0 and 2000.")
            return

        current_high_volume = value
        await message.reply(
            f"✅ High volume set to `{value}/2000`\n"
            f"Multiplier: `{get_high_volume_multiplier(value):.2f}x`"
        )

    except ValueError:
        await message.reply("❌ Enter a valid whole number.")


# ================= COVOLUME =================
@app.on_message(filters.command("covolume", prefixes=".") & sudo_filter)
async def set_covolume(client: Client, message: Message):
    global current_covolume
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(
            f"📣 Co-volume: `{current_covolume}/800`\n"
            f"Multiplier: `{get_covolume_multiplier(current_covolume):.2f}x`"
        )
        return

    try:
        value = int(args[1])
        if not 0 <= value <= 800:
            await message.reply("⚠️ Enter a value between 0 and 800.")
            return

        current_covolume = value
        await message.reply(
            f"✅ Co-volume set to `{value}/800`\n"
            f"Multiplier: `{get_covolume_multiplier(value):.2f}x`\n"
            "⚠️ High output can distort audio."
        )

    except ValueError:
        await message.reply("❌ Enter a valid whole number.")


# ================= ECHO =================
@app.on_message(filters.command("echo", prefixes=".") & sudo_filter)
async def set_echo(client: Client, message: Message):
    global current_echo, echo_buffer
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(f"🔁 Echo: `{current_echo}/10000`")
        return

    try:
        value = int(args[1])
        if not 0 <= value <= 10000:
            await message.reply("⚠️ Echo must be between 0 and 10000.")
            return

        current_echo = value

        if value == 0:
            echo_buffer.fill(0)

        await message.reply(f"✅ Echo set to `{value}/10000`.")

    except ValueError:
        await message.reply("❌ Enter a valid whole number.")


# ================= GAIN =================
@app.on_message(filters.command("gain", prefixes=".") & sudo_filter)
async def set_gain(client: Client, message: Message):
    global current_gain
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(f"🎛 Gain: `{current_gain}/7000`")
        return

    try:
        value = int(args[1])
        if not 0 <= value <= 7000:
            await message.reply("⚠️ Gain must be between 0 and 7000.")
            return

        current_gain = value
        await message.reply(
            f"✅ Gain set to `{value}/7000`.\n"
            "ℹ️ High settings add saturation and may reduce clarity."
        )

    except ValueError:
        await message.reply("❌ Enter a valid whole number.")


# ================= VOICE =================
@app.on_message(filters.command("voice", prefixes=".") & sudo_filter)
async def set_voice(client: Client, message: Message):
    global current_voice, voice_filter_state
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(
            f"🎙 Current voice: `{current_voice}`\n"
            "Usage: `.voice moti`, `.voice patli`, `.voice normal`"
        )
        return

    value = args[1].lower()

    if value not in ("moti", "patli", "normal"):
        await message.reply(
            "⚠️ Choose: `moti`, `patli`, or `normal`."
        )
        return

    current_voice = value
    voice_filter_state.fill(0)

    await message.reply(f"✅ Voice tone set to `{value}`.")


# ================= BASS =================
@app.on_message(filters.command("bass", prefixes=".") & sudo_filter)
async def set_bass(client: Client, message: Message):
    global current_bass, bass_filter_state
    args = (message.text or "").split()

    if len(args) < 2:
        await message.reply(f"🔊 Bass: `{current_bass}/300`")
        return

    try:
        value = int(args[1])
        if not 0 <= value <= 300:
            await message.reply("⚠️ Bass must be between 0 and 300.")
            return

        current_bass = value
        bass_filter_state.fill(0)
        await message.reply(f"✅ Bass set to `{value}/300`.")

    except ValueError:
        await message.reply("❌ Enter a valid whole number.")


# ================= LEAVEVC =================
@app.on_message(filters.command("leavevc", prefixes=".") & sudo_filter)
async def leave_vc(client: Client, message: Message):
    global current_logger_chat, current_target_chat
    global echo_buffer, bass_filter_state, voice_filter_state

    async with bridge_lock:
        if current_logger_chat is None and current_target_chat is None:
            await message.reply("⚠️ No active bridge.")
            return

        chat_ids = {
            chat_id
            for chat_id in (current_logger_chat, current_target_chat)
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
            await message.reply("🔌 Left both voice chats. Bridge stopped.")


# ================= MAIN =================
async def main():
    global USERBOT_ID

    logger.info("Starting Telegram Fight Voice Chat Bot...")
    await app.start()

    try:
        await call_py.start()

        me = await app.get_me()
        USERBOT_ID = me.id

        logger.info(
            "Userbot started as @%s (ID: %s)",
            me.username,
            me.id,
        )

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
