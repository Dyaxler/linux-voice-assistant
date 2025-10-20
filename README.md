# Linux Voice Assistant

Experimental Linux voice assistant for [Home Assistant][homeassistant] that uses the [ESPHome][esphome] protocol.

Runs on Linux `aarch64` and `x86_64` platforms. Tested with Python 3.13 and Python 3.11.
Supports announcments, start/continue conversation, and timers.

## Installation

Install system dependencies (`apt-get`):

* `libportaudio2` or `portaudio19-dev` (for `sounddevice`)
* `build-essential` (for `pymicro-features`)
* `libmpv-dev` (for `python-mpv`)

Clone and install project:

``` sh
git clone https://github.com/OHF-Voice/linux-voice-assistant.git
cd linux-voice-assistant
script/setup
```

## Running

Use `script/run` or `python3 -m linux_voice_assistant`

You must specify `--name <NAME>` with a name that will be available in Home Assistant.

See `--help` for more options.

### Microphone

Use `--audio-input-device` to change the microphone device. Use `python3 -m sounddevice` to see the available PortAudio devices.

The microphone device **must** support 16Khz mono audio.

### Speaker

Use `--audio-output-device` to change the speaker device. Use `mpv --audio-device=help` to see the available MPV devices.

## Wake Word

Wake words are automatically discovered from the directories inside `wakewords/`. Each sub-directory represents a wake word library (for example, `wakewords/microWakeWord` or `wakewords/openWakeWord`). Select the active wake word library and assign models to each assistant from Home Assistant, or edit `preferences.json` to set the defaults (including "No Wake Word"). NOTE: At this time, Home Assistant only allows one active wake word library at a time. I.E. you can't use microWakeWord for Assistant and openWakeWord for Assistant 2.


## Connecting to Home Assistant

1. In Home Assistant, go to "Settings" -> "Device & services"
2. Click the "Add integration" button
3. Choose "ESPHome" and then "Set up another instance of ESPHome"
4. Enter the IP address of your voice satellite with port 6053
5. Click "Submit"

<!-- Links -->
[homeassistant]: https://www.home-assistant.io/
[esphome]: https://esphome.io/
