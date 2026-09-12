# 🎹 MIDI Editor - Web-based Singing MIDI Editor

[English](README.md) | [简体中文](README_CN.md)

A full-featured web MIDI editor for singing voice preprocess. It supports real-time drag editing of MIDI notes, lyric editing, audio waveform alignment, and importing/exporting MIDI files with lyrics.

![MIDI Editor](https://img.shields.io/badge/React-19.2-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue) ![Vite](https://img.shields.io/badge/Vite-7.2-purple)

## ✨ Features

### 🎼 Piano Roll Editing

- **Visual note editing**: Full range from C1 to C8 with intuitive piano key layout
- **Drag operations**:
  - Move notes: drag note blocks to adjust position and pitch
  - Resize start: drag the left edge to adjust start time
  - Resize end: drag the right edge to adjust end time
- **Quick pitch adjust**:
  - Command/Ctrl + Up/Down to nudge selected note pitch
  - Use the Transpose control in the toolbar to shift all notes at once
- **Double-click to add**: Add new notes quickly in empty areas
- **Piano key preview**: Click a key on the left to audition the pitch

### 🔍 Zoom & Navigation

- **Horizontal zoom**
- **Vertical zoom**
- **Dynamic snapping**: finer snap granularity at higher zoom (min 0.01s)
- **Auto scroll**: keep the playhead visible during playback

### 📝 Lyric Editing

- **Inline editing**: edit lyrics for each note in the side list
- **Batch fill**: enter lyrics and auto-fill notes in order
- **Fill from selection**: start batch fill from the currently selected note
- **Precise fields**: edit PITCH, START, and END directly
- **Confirm edits**: press Enter or click ✓ to confirm, avoiding accidental changes

### 🎵 Audio Alignment

- **Waveform display**: import audio to display waveform, synced with the MIDI timeline
- **Formats**: MP3, WAV, OGG, FLAC, M4A, AAC
- **Sync playback**: play audio and MIDI together with independent volume control
- **Click to seek**: click waveform or timeline to seek

### ⚠️ Overlap Detection

- **Visual highlight**: overlapping notes blink in red
- **One-click fix**: remove all overlaps automatically

### 📥 Import & Export

- **MIDI import**: parse standard MIDI files with automatic lyric metadata extraction
- **MIDI export**: export MIDI files with lyric information

### 🎨 UI & UX

- **Theme toggle**: light and dark modes
- **Responsive layout**: adapts to window size
- **SVG grid**: cross-browser grid rendering
- **Status feedback**: real-time state and error tips

## 🚀 Quick Start

### Requirements

- Node.js 18+
- npm or yarn

### Install

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Expose to LAN
npm run dev -- --host 0.0.0.0
```

### Build

```bash
# Build for production
npm run build

# Preview build
npm run preview
```

## 📖 Usage

### Basic Workflow

1. **Import MIDI**: click Import MIDI and select a .mid file
2. **Edit notes**: drag notes in the piano roll to adjust time and pitch
3. **Add lyrics**: edit lyrics in the right-side list, or use batch fill
4. **Align audio** (optional): import reference audio for side-by-side editing
5. **Export**: click Export MIDI to save

### Shortcuts

| Action | Description |
|------|------|
| Double-click piano roll | Add a new note |
| Double-click note | Edit lyric |
| Drag note | Move note and pitch |
| Drag note edges | Resize note |
| Backspace / Delete | Delete selected note |
| Enter | Confirm value edits |
| Escape | Cancel value edits |
| Ctrl(Command) + Wheel | Horizontal zoom |
| Ctrl(Command) + Shift(Option) + Wheel | Vertical zoom |

### Playback Controls

| Button | Description |
|------|------|
| ⏮ | Go to start |
| ⏪ | Back 2 seconds |
| ▶ / ⏸ | Play / Pause |
| ⏩ | Forward 2 seconds |
| ⏭ | Go to end |
| Selection | Play selected region |

## 🛠 Tech Stack

- **Frontend**: React 19 + TypeScript
- **Build**: Vite 7
- **State**: Zustand
- **Audio**: Tone.js
- **Waveform**: WaveSurfer.js
- **MIDI**: @tonejs/midi
- **Styles**: CSS with custom variables

## 📁 Project Structure

```
.
├── eslint.config.js
├── index.html
├── package.json
├── postcss.config.js
├── README.md
├── README_CN.md
├── tailwind.config.js
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── public/
└── src/
    ├── App.css              # Main styles (theme variables, layout, components)
    ├── App.tsx               # Main app component (transport, import/export, transpose)
    ├── constants.ts          # Constants (grid width, row height, pitch range)
    ├── i18n.ts               # Internationalization (zh/en translations, smart lyric tokenizer)
    ├── index.css             # Global styles (Tailwind, root font, theme gradients)
    ├── main.tsx              # React entry point
    ├── types.ts              # Type definitions (NoteEvent, TimeSignature, etc.)
    ├── components/
    │   ├── AudioTrack.tsx    # Audio waveform display component
    │   ├── LyricTable.tsx    # Lyric editing table component
    │   └── PianoRoll.tsx     # Piano roll editor component
    ├── lib/
    │   └── midi.ts           # MIDI import/export utilities (UTF-8 lyric encoding)
    └── store/
        └── useMidiStore.ts   # Zustand state management
```
