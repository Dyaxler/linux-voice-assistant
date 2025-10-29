# Linux Voice Assistant

Experimental Linux voice assistant for [Home Assistant][homeassistant] that uses the [ESPHome][esphome] protocol.

Runs on Linux `aarch64` and `x86_64` platforms. Tested with Python 3.13 and Python 3.11.

## Features

* Native [ESPHome Voice][esphome] transport for fast, local command handling.
* Multi wake word support with per-assistant model selection and sensitivity tuning.
* Includes "stop word" handling to cancel active audio or assist pipelines. Just say "stop" at anytime.
* Distinct media and TTS playback paths with automatic ducking for announcements.
* Announcements, start/continue conversation flows, and full timer lifecycle events.
* Configuration entities exposed to Home Assistant for wake word library, mute toggle, and preference management.

## Installation

### System requirements

* 64-bit Linux (Debian 12/13, Ubuntu 22.04+, Fedora 39+, etc.).
* Python 3.11 or 3.13. Earlier versions are not supported.
* Audio stack capable of 16 kHz mono input and standard PCM output.
* Network connectivity to Home Assistant's ESPHome integration.

Install system dependencies (`apt-get` or the equivalent for your distribution):

* `libportaudio2` or `portaudio19-dev` (microphone access via `sounddevice`).
* `build-essential` (build tools required by `pymicro-features`).
* `libmpv-dev` (shared libraries for `python-mpv`).
* `python3-venv` (recommended for isolated deployments).

Clone and install the project:

``` sh
git clone https://github.com/OHF-Voice/linux-voice-assistant.git
cd linux-voice-assistant
script/setup
```

The `script/setup` helper creates a virtual environment, installs project dependencies, and compiles bundled wake word models. To reuse an existing Python environment, run `pip install .` instead.

## Running

Use `script/run` or `python3 -m linux_voice_assistant`

You must specify `--name <NAME>` with a name that will be available in Home Assistant.

See `--help` for more options.

### Microphone

Use `--audio-input-device` to change the microphone device. Use `python3 -m sounddevice` to see the available PortAudio devices.
(Occasionally you'll need to use the full path: `/opt/linux-voice-assistant/.venv/bin/python3 -m sounddevice`).

The microphone device **must** support 16Khz mono audio.

### Speaker

Use `--audio-output-device` to change the speaker device. Use `mpv --audio-device=help` to see the available MPV devices.

### Service configuration

For unattended deployments (for example, on a Raspberry Pi kiosk) install the assistant as a systemd service. The snippet below assumes the project lives at `/opt/linux-voice-assistant` inside a virtual environment:

```ini
[Unit]
Description=Linux Voice Assistant
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=voice
WorkingDirectory=/opt/linux-voice-assistant
Environment="PATH=/opt/linux-voice-assistant/.venv/bin"
ExecStart=/opt/linux-voice-assistant/.venv/bin/python -m linux_voice_assistant --name "Living Room Assistant"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

After saving the unit file run `sudo systemctl daemon-reload` followed by `sudo systemctl enable --now linux-voice-assistant.service`.

## Wake Word

Wake words are automatically discovered from the directories inside `wakewords/`. Each sub-directory represents a wake word library (for example, `wakewords/microWakeWord` or `wakewords/openWakeWord`). Select the active wake word library and assign models to each assistant from Home Assistant, or edit `preferences.json` to set the defaults (including "No Wake Word"). NOTE: At this time, Home Assistant only allows one active wake word library at a time. I.E. you can't use microWakeWord for Assistant and openWakeWord for Assistant 2.

### Wake word management tips

* Preferences are persisted to `<config>/preferences.json` (typically under the project root). Delete the file to reset wake word selections.
* Use the `Wake Word Library` select entity in Home Assistant to switch libraries. The integration refreshes models automatically, applying your preferred sensitivity.
* Multi wake word support lets you assign different models to the primary and secondary assistants. Use the `Wake Word Sensitivity` select and the `Wake Sound`/`Timer Sound` switches to fine tune behavior.

### Stop word behavior

If a library contains a `stop.json` model it is loaded as the "stop word". Saying the configured phrase cancels the current pipeline, wakes, or timers after a short cooldown. Review the `wakewords/<library>/stop.json` file for the exact trigger phrase and adjust the probability threshold in `preferences.json` if necessary. Currently there is only a microWakeWord stop model available. It's loaded by default and works with either wake word libraries.


## Connecting to Home Assistant

1. In Home Assistant, go to "Settings" -> "Device & services"
2. Click the "Add integration" button
3. Choose "ESPHome" and then "Set up another instance of ESPHome"
4. Enter the IP address of your voice satellite with port 6053
5. Click "Submit"

NOTE: If ESPHome is already installed then once the server (ran manually via CLI) or the service starts, the new satellite should be auto detected and give you the options to either add or ignore. It'll be identified by the name you specify in the cmdline arguments `--name <NAME>`.

Once connected you should see entities for the media player, mute switch, wake word library select, reset/restart buttons, and optional configuration switches (wake sound, timer sound, and wake word sensitivity). Use Home Assistant's logbook to verify multi wake word triggers and timer events.

## Troubleshooting

* **No audio input detected** – confirm the microphone reports a 16 kHz mono capable mode via `python3 -m sounddevice` (Occasionally you'll need to use the full path: `/opt/linux-voice-assistant/.venv/bin/python3 -m sounddevice`). Configure ALSA/PulseAudio so that the selected device is not in exclusive use by another process.
* **Audio output too loud or quiet** – adjust the media player's volume entity in Home Assistant or change the hardware volume mixer. Changes persist to `preferences.json`.
* **Service fails to start** – check `journalctl -u linux-voice-assistant.service` for errors. Missing `libmpv` or PortAudio dependencies are the most common causes. You can edit your service file and add the `--debug` argument to the end of your ExecStart line. This will provide some additional insight.
* **Wake words not listed** – verify the wake word directory structure is `wakewords/<library>/<model>.json`. Use `script/setup` after adding new models so that compiled assets are generated. Restarting the server/service pretty much does the same thing but running `script/setup` makes sure no files are corrupt.
* **ESPHome integration cannot connect** – ensure port 6053 is reachable and that the assistant is started with a unique `--name`. Duplicate names or conflicting ESPHome nodes will prevent discovery. Make sure you're surrounding the `--name` value in double quotes `"Hello World Satellite"` if you want your satellite name to contain spaces.

See [`CHANGELOG.md`](CHANGELOG.md) for recent enhancements including multi wake word support, stop word improvements, and wake pipeline refinements introduced after `1.0.0`.

<!-- Links -->
[homeassistant]: https://www.home-assistant.io/
[esphome]: https://esphome.io/
