"""Engine-specific TTS script optimisation profiles.

This module is the single source of truth for per-engine text preparation guidance in
Ultimate TTS Studio.  It serves two consumers:

1. **Coding agents** — read this file as a skill/reference document when writing or
   reviewing code that prepares text for a specific TTS engine.

2. **Runtime narration transform pipeline** — ``get_engine_prompt_addendum()`` and
   ``get_engine_script_rules()`` are called by ``narration_transform.py`` to append
   engine-specific instructions to the LLM system prompt before each transform run.

Engine names in ``ENGINE_SCRIPT_PROFILES`` match the keys used in
``engine_registry.ENGINE_EXPRESSIVENESS`` and in the UI engine dropdown.

No imports from other app modules are required; this file is intentionally self-contained.
"""

from __future__ import annotations

ENGINE_SCRIPT_PROFILES: dict[str, dict] = {
    # ------------------------------------------------------------------ ChatterboxTTS
    "ChatterboxTTS": {
        "engine_id": "chatterbox",
        "display_name": "🗣️ ChatterboxTTS",
        "script_rules": [
            "Keep sentences under 250 characters; the engine's internal chunker breaks at "
            "natural pause points — short sentences improve prosody consistency.",
            "Use ellipsis ('...') for deliberate mid-sentence pauses; commas create lighter "
            "micro-pauses.",
            "Em-dashes ('—') work well for dramatic pivots or interrupted thoughts.",
            "Avoid SSML tags, bracket cues such as [laughing], or parenthetical stage directions "
            "(smiling) — the engine strips them and may mis-pronounce the surrounding text.",
            "Spell out abbreviations, acronyms, and initialisms in full spoken form unless they "
            "are common (NATO, NASA, AI).",
            "Normalise numbers to words; mixed digit/word strings cause rhythm artefacts.",
            "When voice cloning is active, write a reference phrase whose rhythm and register "
            "match the target voice before the main script.",
            "Avoid exclamation marks in fast or technical passages — the engine can over-stress "
            "them unpredictably.",
        ],
        "avoid": [
            "SSML tags (<break>, <phoneme>, <prosody>, etc.)",
            "Bracket stage directions: [sighing], [whisper], [fast]",
            "Parenthetical asides that read as stage directions: (nervously), (loud)",
            "All-caps emphasis words in dense passages — the exaggeration slider handles stress",
            "Long unbroken sentences over 300 characters",
        ],
        "prompt_addendum": (
            "You are optimising text for ChatterboxTTS, a neural voice-cloning TTS engine that "
            "uses autoregressive generation with an exaggeration control slider.  "
            "Keep sentences short to medium (under 250 characters) so the engine's internal "
            "chunker can create clean prosody boundaries.  "
            "Use ellipsis ('...') for pauses and em-dashes ('—') for dramatic breaks.  "
            "Spell out all numbers, currencies, and abbreviations in full spoken form.  "
            "Do NOT emit SSML tags, bracket cues, or parenthetical stage directions — the engine "
            "does not interpret them and they degrade output quality.  "
            "All-caps emphasis is not needed; expressive delivery is controlled by the slider."
        ),
        "max_chunk_chars": 250,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 24000,
        "notes": (
            "Exaggeration (0–2) and cfg_weight (0–1) shape delivery; higher exaggeration "
            "benefits from more dramatic punctuation in the script.  24 kHz output."
        ),
    },
    # -------------------------------------------------------- Chatterbox Multilingual
    "Chatterbox Multilingual": {
        "engine_id": "chatterbox_multilingual",
        "display_name": "🌍 Chatterbox Multilingual",
        "script_rules": [
            "Write text in the target language only — do not mix languages in a single chunk.",
            "Keep sentences under 250 characters; multilingual autoregressive models degrade "
            "on long sequences.",
            "Use the locale's native punctuation conventions for pauses "
            "(e.g., '...' universally, but language-specific dash conventions).",
            "Spell out numbers, dates, and currency in the target language's spoken form.",
            "Avoid SSML tags and bracket stage directions — unsupported.",
            "For languages with diacritics (French, Spanish, German), preserve them exactly; "
            "the model uses them for correct pronunciation.",
            "Repetition-penalty artefacts are more visible in multilingual mode — avoid "
            "repeating the same word or phrase more than twice in one chunk.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical cues",
            "Mixed-language sentences in a single chunk",
            "Digit-only numbers (write out in target language)",
        ],
        "prompt_addendum": (
            "You are optimising text for Chatterbox Multilingual, a multilingual neural TTS "
            "engine that supports voice cloning across languages.  "
            "Write sentences in the target language only; do not mix languages within a chunk.  "
            "Keep sentences under 250 characters.  "
            "Spell out numbers, dates, and currencies in the native spoken form of the locale.  "
            "Preserve diacritics exactly — they drive correct phoneme selection.  "
            "Do NOT emit SSML tags or bracket/parenthetical stage directions; they are not "
            "interpreted and will degrade output.  "
            "Avoid repeating the same word more than twice per chunk to prevent repetition "
            "penalty artefacts."
        ),
        "max_chunk_chars": 250,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 24000,
        "notes": (
            "Language parameter is set at inference time but the script must be in the chosen "
            "language.  repetition_penalty, min_p, top_p sliders provide additional control."
        ),
    },
    # ----------------------------------------------------------- Chatterbox Turbo
    "Chatterbox Turbo": {
        "engine_id": "chatterbox_turbo",
        "display_name": "⚡ Chatterbox Turbo",
        "script_rules": [
            "Keep sentences under 200 characters — Turbo's distilled decoder is optimised for "
            "shorter inputs and prosody quality drops on longer sequences.",
            "Use ellipsis ('...') for pauses; commas for light rhythm breaks.",
            "Em-dash ('—') signals a strong pivot or interruption.",
            "Avoid SSML tags, bracket cues, and parenthetical stage directions.",
            "Spell out numbers and abbreviations; Turbo inherits the same tokeniser as the base "
            "model and numeric strings cause the same rhythm artefacts.",
            "Prefer active-voice, declarative sentences — Turbo's distillation is most stable "
            "with clear syntactic structure.",
            "If using voice cloning, the reference audio duration should be 5–15 seconds of "
            "clean speech; script phrasing should mirror the reference's register.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Sentences over 300 characters",
            "Heavy all-caps passages",
        ],
        "prompt_addendum": (
            "You are optimising text for Chatterbox Turbo (SUP3RMASS1VE/turbo-chatterbox), a "
            "fast distilled version of ChatterboxTTS.  "
            "Keep sentences under 200 characters for best prosody; very short sentences (under "
            "80 characters) are ideal for this engine.  "
            "Use ellipsis for deliberate pauses and em-dashes for dramatic pivots.  "
            "Spell out all numbers and abbreviations in spoken form.  "
            "Do NOT emit SSML tags or any bracket/parenthetical cues — not supported.  "
            "Prefer simple, declarative sentence structure; the distilled decoder is most "
            "consistent with clear syntax."
        ),
        "max_chunk_chars": 200,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 24000,
        "notes": (
            "Fastest Chatterbox variant.  Same parameter surface as Chatterbox Multilingual "
            "(exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, "
            "chunk_size).  24 kHz output."
        ),
    },
    # ------------------------------------------------------------------ Kokoro TTS
    "Kokoro TTS": {
        "engine_id": "kokoro",
        "display_name": "🗣️ Kokoro TTS",
        "script_rules": [
            "Keep sentences under 200 characters for optimal prosody; Kokoro is a fast ONNX "
            "model that performs best with shorter input spans.",
            "Use ellipsis ('...') for mid-sentence pauses; Kokoro responds well to this cue.",
            "Commas produce light rhythmic breaks; semicolons produce medium pauses.",
            "Spell out numbers, ordinals, currencies, dates, and abbreviations in spoken form "
            "(the engine does no text normalisation internally).",
            "Avoid SSML, bracket cues, and parenthetical stage directions — not parsed.",
            "All-caps emphasis is not interpreted; stress is controlled by the speed slider.",
            "Kokoro supports 34 built-in voices across US English, UK English, Portuguese, and "
            "Italian — locale matching in the script is important (use the correct idioms and "
            "spellings for the selected voice).",
            "For US voices, write American English spelling; for UK voices, write British "
            "English spelling.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical cues",
            "Unspoken digits and special characters (%, $, &, @)",
            "Run-on sentences over 250 characters",
        ],
        "prompt_addendum": (
            "You are optimising text for Kokoro TTS, a lightweight ONNX neural TTS engine with "
            "34 built-in voices (US English, UK English, Portuguese, Italian).  "
            "Keep sentences short to medium (under 200 characters).  "
            "Use ellipsis ('...') for pauses and standard punctuation for rhythm; Kokoro has "
            "no special tag system.  "
            "Spell out every number, currency symbol, percentage, date, and abbreviation in "
            "fully spoken form — the engine performs no internal normalisation.  "
            "Match locale spelling and idioms to the selected voice's language variant.  "
            "Do NOT emit SSML tags or bracket/parenthetical stage cues."
        ),
        "max_chunk_chars": 500,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": False,
        "sample_rate_hz": 24000,
        "notes": (
            "Very fast ONNX inference.  Speed slider (0.5–2.0x) adjusts delivery pace.  "
            "34 voices: US (af_heart, af_bella, af_nicole, …, am_michael, …), "
            "UK (bf_emma, bm_george, …), Portuguese (pf_dora, pm_alex), Italian (if_sara, "
            "im_nicola).  No voice cloning."
        ),
    },
    # ------------------------------------------------------------------ Fish Speech
    "Fish Speech": {
        "engine_id": "fish_speech",
        "display_name": "🐟 Fish Speech",
        "script_rules": [
            "Fish Speech operates as an autoregressive language-model TTS — longer, "
            "well-structured sentences produce natural prosody.",
            "Sentences of 50–300 characters work well; very short staccato sentences can "
            "sound choppy without a voice reference.",
            "Use ellipsis ('...') and em-dashes ('—') for pacing.",
            "Avoid SSML tags and bracket stage directions.",
            "Spell out numbers, abbreviations, and special symbols in spoken form.",
            "With voice cloning: provide a reference audio matching the desired register; "
            "write text whose prosodic rhythm matches the reference audio.",
            "Repetition-penalty artefacts appear as word echo loops — avoid repeating the "
            "same phrase more than twice within a paragraph.",
            "Fish Speech has good multilingual capability; always write text in the target "
            "language rather than mixing languages in a single segment.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Identical phrase repetition within 200 characters (triggers repetition penalty "
            "artefact)",
        ],
        "prompt_addendum": (
            "You are optimising text for Fish Speech, an autoregressive LM-based TTS engine "
            "with strong multilingual and voice-cloning capabilities.  "
            "Write well-formed sentences of 50–300 characters.  "
            "Use ellipsis for pauses and em-dashes for pivots.  "
            "Spell out numbers, abbreviations, and special symbols in spoken form.  "
            "Do NOT emit SSML tags or bracket/parenthetical stage cues.  "
            "Avoid repeating the same word or phrase more than twice within a single chunk — "
            "this triggers the repetition penalty and causes echo artefacts.  "
            "When multiple languages are involved, write each chunk in a single language only."
        ),
        "max_chunk_chars": 300,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 44100,
        "notes": (
            "temperature, top_p, repetition_penalty, max_tokens are exposed in the UI.  "
            "Multilingual out of the box.  Reference audio improves voice consistency.  "
            "44.1 kHz output."
        ),
    },
    # ------------------------------------------------------------------- IndexTTS
    "IndexTTS": {
        "engine_id": "indextts",
        "display_name": "📑 IndexTTS",
        "script_rules": [
            "IndexTTS is an autoregressive voice-cloning engine — use clear, natural sentence "
            "structure for best cloning fidelity.",
            "Keep segments under 300 characters; very long inputs can cause timing drift.",
            "Use ellipsis ('...') for pauses and em-dashes ('—') for strong breaks.",
            "Spell out numbers, abbreviations, and special characters in spoken form.",
            "Avoid SSML tags and bracket stage directions — not supported.",
            "Reference audio quality (clean, minimal background noise) is more important than "
            "script style for this engine.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Unspoken digits or symbols",
        ],
        "prompt_addendum": (
            "You are optimising text for IndexTTS, a voice-cloning TTS engine based on "
            "autoregressive generation.  "
            "Write clean, natural sentences under 300 characters.  "
            "Use ellipsis for deliberate pauses and em-dashes for breaks.  "
            "Spell out all numbers, abbreviations, and symbols in spoken form.  "
            "Do NOT emit SSML tags or bracket/parenthetical cues.  "
            "The engine derives pacing and tone from the reference audio, so the script should "
            "be clean and well-formed to let the cloned voice come through clearly."
        ),
        "max_chunk_chars": 300,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 22050,
        "notes": (
            "Temperature slider is the only generation parameter exposed.  "
            "22.05 kHz output.  Voice cloning only (no built-in voices)."
        ),
    },
    # ------------------------------------------------------------------ IndexTTS2
    "IndexTTS2": {
        "engine_id": "indextts2",
        "display_name": "📑🆕 IndexTTS2",
        "script_rules": [
            "IndexTTS2 supports bracket cues AND fine-grained emotion vectors — this is the "
            "richest expressiveness surface in the studio.",
            "For text_description emotion mode, write a bracket cue at the start of the "
            "sentence to set the emotional register: "
            "[happy], [sad], [angry], [afraid], [disgusted], [melancholic], [surprised], "
            "[calm].",
            "Only ONE bracket cue per sentence; stacking cues in the same sentence is not "
            "supported.",
            "Keep sentences under 200 characters; max_mel_tokens caps synthesis length — "
            "longer sentences may be cut off.",
            "Use ellipsis ('...') for dramatic pauses; em-dashes ('—') for pivots.",
            "Spell out numbers, abbreviations, and symbols in spoken form.",
            "Avoid SSML tags — not supported.",
            "In vector_control mode, the UI sliders set emotion; no bracket cues are needed "
            "in the script.",
            "In audio_reference mode, the reference audio drives emotion; bracket cues are "
            "optional but can reinforce intent.",
        ],
        "avoid": [
            "SSML tags (<break>, <phoneme>, <prosody>, etc.)",
            "Multiple bracket cues in the same sentence",
            "Unknown bracket labels (only the 8 listed emotion labels are recognised)",
            "Sentences over 250 characters (risk of mel-token overflow at default 1500 limit)",
        ],
        "prompt_addendum": (
            "You are optimising text for IndexTTS2, the most expressive engine in the studio.  "
            "IndexTTS2 supports bracket-style emotion cues: place ONE of [happy], [sad], "
            "[angry], [afraid], [disgusted], [melancholic], [surprised], or [calm] at the "
            "start of a sentence to set emotional register when using text_description emotion "
            "mode.  Do NOT stack multiple bracket cues on the same sentence.  "
            "Keep sentences under 200 characters to avoid mel-token overflow.  "
            "Use ellipsis ('...') for deliberate pauses and em-dashes ('—') for pivots.  "
            "Spell out all numbers, dates, currencies, and abbreviations in spoken form.  "
            "Do NOT emit SSML tags — they are not supported.  "
            "If the user's emotion mode is vector_control or audio_reference, omit bracket "
            "cues from the script entirely — the UI sliders or reference audio control emotion."
        ),
        "max_chunk_chars": 200,
        "supports_emotion_tags": True,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 22050,
        "notes": (
            "Three emotion modes: audio_reference (default), vector_control (8-axis sliders), "
            "text_description (bracket cues in script).  "
            "Powered by IndexTTS-2 + Qwen0.6B emotion model.  "
            "22.05 kHz output.  max_mel_tokens default: 1500."
        ),
    },
    # -------------------------------------------------------------------- F5-TTS
    "F5-TTS": {
        "engine_id": "f5_tts",
        "display_name": "🎵 F5-TTS",
        "script_rules": [
            "F5-TTS is a flow-matching voice-cloning engine — prosody tracks the reference "
            "audio very closely; keep script tone consistent with the reference.",
            "Sentences of 50–250 characters work best; very short isolated words sound "
            "clipped without context.",
            "Use ellipsis ('...') for pauses and em-dashes ('—') for dramatic breaks.",
            "Spell out all numbers, abbreviations, and special characters in spoken form.",
            "Avoid SSML tags and bracket stage directions — not supported.",
            "For the language-specific model variants (French, German, Japanese, Spanish), "
            "always write text in the target language.",
            "The remove_silence option in the UI removes trailing silences; if pauses are "
            "intentional, use punctuation rather than relying on trailing silence.",
            "cross_fade duration (UI slider) controls blending between generated audio chunks; "
            "natural sentence boundaries in the script help cross-fade sound seamless.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Mixed-language text in a single chunk when using language-specific models",
        ],
        "prompt_addendum": (
            "You are optimising text for F5-TTS, a flow-matching zero-shot voice-cloning TTS "
            "engine.  F5-TTS derives delivery from the reference audio, so the script should "
            "match the register and tone of the reference.  "
            "Keep sentences 50–250 characters.  "
            "Use ellipsis for pauses; avoid relying on trailing silence (the remove_silence "
            "option may eliminate it).  "
            "Spell out all numbers, abbreviations, and special characters in spoken form.  "
            "Do NOT emit SSML tags or bracket/parenthetical cues.  "
            "For language-specific model variants (French, German, Japanese, Spanish), "
            "write text only in the target language."
        ),
        "max_chunk_chars": 250,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 24000,
        "notes": (
            "Available model variants: F5-TTS Base, F5-TTS v1 Base, plus fine-tunes for "
            "French, German, Japanese, Spanish.  "
            "speed, cross_fade, and remove_silence sliders in the UI.  24 kHz output."
        ),
    },
    # ---------------------------------------------------------------- Higgs Audio
    "Higgs Audio": {
        "engine_id": "higgs_audio",
        "display_name": "🔊 Higgs Audio",
        "script_rules": [
            "Higgs Audio has a HARD chunk limit of 100 characters per segment — the handler "
            "splits text at sentence boundaries and then at word boundaries to enforce this.  "
            "Write many short sentences (under 100 characters) to avoid mid-word splits.",
            "Each sentence should be self-contained because chunking at the character level "
            "means context does not carry across chunks.",
            "Use clear sentence-ending punctuation (.  !  ?) so the splitter can find good "
            "break points.",
            "Ellipsis ('...') signals a pause but counts toward the 100-character limit.",
            "Spell out all numbers, abbreviations, and special characters.",
            "Avoid SSML tags and bracket stage directions — not supported.",
            "Higgs Audio exposes a system_prompt UI field; the narration transform should "
            "target coherent short sentences, not add style cues, because the system prompt "
            "already sets the voice's character.",
            "All-caps words may cause shouting artefacts — capitalise only proper nouns and "
            "sentence starts.",
        ],
        "avoid": [
            "Sentences over 100 characters (will be force-split mid-word)",
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "All-caps emphasis words",
            "Complex nested clauses (they cross chunk boundaries awkwardly)",
        ],
        "prompt_addendum": (
            "You are optimising text for Higgs Audio, an LLM-style TTS engine with a HARD "
            "100-character chunk limit.  "
            "CRITICAL: every sentence MUST be under 100 characters (including spaces and "
            "punctuation).  Prefer 60–90 characters per sentence.  "
            "Write many short, self-contained sentences — context does not carry across the "
            "chunk boundary.  "
            "End every sentence with clear punctuation (period, exclamation, question mark) "
            "so the chunker finds clean break points.  "
            "Spell out all numbers, abbreviations, and special characters in spoken form.  "
            "Use lowercase for emphasis — all-caps words can cause shouting artefacts.  "
            "Do NOT emit SSML tags or bracket/parenthetical cues."
        ),
        "max_chunk_chars": 100,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": False,
        "sample_rate_hz": 24000,
        "notes": (
            "LLM-style text-to-speech model with voice presets.  "
            "Exposes a system_prompt UI field for voice character instructions.  "
            "RAS (repetition-aware sampling) parameters: ras_win_len, ras_win_max_num_repeat.  "
            "Hard 100-char chunk limit enforced by the handler."
        ),
    },
    # -------------------------------------------------------------------- VoxCPM
    "VoxCPM": {
        "engine_id": "voxcpm",
        "display_name": "🎙️ VoxCPM",
        "script_rules": [
            "VoxCPM1.5 uses a 44.1 kHz Audio VAE — it produces broadcast-quality audio when "
            "fed clean, well-punctuated text.",
            "Sentences of 50–300 characters work well; the model handles longer inputs but "
            "prosody quality is best in this range.",
            "Use ellipsis ('...') for pauses and em-dashes ('—') for dramatic breaks.",
            "Spell out all numbers, abbreviations, and special characters in spoken form.",
            "Avoid SSML tags and bracket stage directions — not supported.",
            "VoxCPM supports a conversation mode with per-speaker seed tracking; in "
            "multi-speaker scripts, label each line with the speaker name (e.g. 'Alice:') so "
            "the pipeline can route consistently.",
            "The engine respects speaker attribution labels — preserve 'SPEAKER_NAME:' "
            "prefixes exactly as written.",
            "Avoid over-stressing words with all-caps — the model's high sample rate means "
            "artefacts are very audible.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "All-caps emphasis in dense technical passages",
            "Unspoken digits and symbols",
        ],
        "prompt_addendum": (
            "You are optimising text for VoxCPM (openbmb/VoxCPM1.5), a high-fidelity "
            "voice-cloning TTS engine outputting 44.1 kHz audio.  "
            "Write clean, well-punctuated sentences of 50–300 characters.  "
            "Use ellipsis for deliberate pauses and em-dashes for pivots.  "
            "Spell out all numbers, abbreviations, and special characters.  "
            "If the script has multiple speakers, preserve 'SpeakerName:' prefixes exactly — "
            "the engine uses per-speaker seed tracking in conversation mode.  "
            "Do NOT emit SSML tags or bracket/parenthetical cues.  "
            "Avoid all-caps emphasis; the engine's high output fidelity makes artefacts very "
            "noticeable."
        ),
        "max_chunk_chars": 300,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 44100,
        "notes": (
            "VoxCPM1.5 model from openbmb.  44.1 kHz Audio VAE output.  "
            "Conversation mode with per-speaker seed consistency.  "
            "Whisper is used internally for reference-audio transcription."
        ),
    },
    # ----------------------------------------------------------------- KittenTTS
    "KittenTTS": {
        "engine_id": "kitten_tts",
        "display_name": "🐱 KittenTTS",
        "script_rules": [
            "KittenTTS is an ultra-lightweight ONNX model (KittenML/kitten-tts-mini-0.1) — "
            "optimise for clean, simple narration with short sentences.",
            "Keep sentences under 150 characters; the mini model performs best on concise "
            "inputs.",
            "Use ellipsis ('...') for pauses and commas for light rhythm breaks.",
            "Spell out all numbers, abbreviations, and special characters.",
            "Avoid SSML tags, bracket stage directions, and complex nested clauses.",
            "The 8 built-in voices (expr-voice-2 through expr-voice-5, male and female) "
            "have fixed expressive profiles — script tone should match the chosen voice's "
            "character.",
            "All-caps emphasis is not interpreted; keep capitalisation standard.",
            "Short, punchy declarative sentences work best for this engine.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Sentences over 200 characters",
            "All-caps emphasis",
            "Complex multi-clause sentences",
        ],
        "prompt_addendum": (
            "You are optimising text for KittenTTS, an ultra-lightweight ONNX TTS model with "
            "8 built-in expressive voices.  "
            "Write very short, clean sentences (under 150 characters).  "
            "Use ellipsis for pauses and commas for light breaks.  "
            "Spell out all numbers, abbreviations, and special characters.  "
            "Prefer simple, declarative sentence structure — the mini model is most consistent "
            "with short, clear inputs.  "
            "Do NOT emit SSML tags or bracket/parenthetical stage cues.  "
            "All-caps emphasis is not interpreted by this engine."
        ),
        "max_chunk_chars": 150,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": False,
        "sample_rate_hz": 24000,
        "notes": (
            "Model: KittenML/kitten-tts-mini-0.1.  ONNX inference.  8 voices: "
            "expr-voice-{2,3,4,5}-{m,f}.  GPU acceleration via CUDAExecutionProvider if "
            "onnxruntime-gpu is installed.  Very fast.  24 kHz output."
        ),
    },
    # ---------------------------------------------------------- Qwen Voice Design
    "Qwen Voice Design": {
        "engine_id": "qwen_voice_design",
        "display_name": "✏️ Qwen Voice Design",
        "script_rules": [
            "Qwen Voice Design creates a custom voice FROM a natural-language description — "
            "the script should be representative of how that voice will be used.",
            "Write in the same register and language as the voice description.",
            "Sentences of 50–200 characters; Qwen3-TTS-VoiceDesign-1.7B does not support "
            "chunked batch processing.",
            "Use ellipsis ('...') for pauses and em-dashes ('—') for breaks.",
            "Spell out all numbers, abbreviations, and special characters.",
            "Avoid SSML tags and bracket stage directions.",
            "The voice is created at inference time — there is no reference audio; the "
            "LLM-driven voice design handles timbre and style.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Very long sentences (over 250 characters)",
            "Language-switching mid-sentence",
        ],
        "prompt_addendum": (
            "You are optimising text for Qwen Voice Design (Qwen3-TTS-VoiceDesign-1.7B), which "
            "synthesises speech using a natural-language voice description rather than a "
            "reference audio.  "
            "Write sentences in the same language and register as the voice description.  "
            "Keep sentences 50–200 characters; this mode does not chunk long inputs.  "
            "Use ellipsis for pauses and em-dashes for pivots.  "
            "Spell out all numbers, abbreviations, and special characters.  "
            "Do NOT emit SSML tags or bracket/parenthetical cues.  "
            "The voice's character is determined by the design prompt, not by script cues, so "
            "write clean natural narration without trying to add style instructions in the text."
        ),
        "max_chunk_chars": 200,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": False,
        "sample_rate_hz": 24000,
        "notes": (
            "Model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign.  Voice is described in natural "
            "language via the UI.  No chunking support — single-pass generation.  "
            "Supported languages: Chinese, English, Japanese, Korean, French, German, Spanish, "
            "Portuguese, Russian."
        ),
    },
    # ---------------------------------------------------------- Qwen Voice Clone
    "Qwen Voice Clone": {
        "engine_id": "qwen_voice_clone",
        "display_name": "🎤 Qwen Voice Clone",
        "script_rules": [
            "Qwen Voice Clone (Base model) performs voice cloning from reference audio with "
            "chunked generation up to 200 characters per chunk.",
            "Write sentences whose rhythm matches the reference audio register; Qwen's "
            "autoregressive decoder tracks pacing from context.",
            "Keep individual sentences under 200 characters — the handler's chunk_text() "
            "splits at sentence boundaries then at words up to this limit.",
            "Use ellipsis ('...') for pauses and em-dashes ('—') for breaks.",
            "Spell out all numbers, abbreviations, and special characters.",
            "Avoid SSML tags and bracket stage directions.",
            "Avoid repeating the same phrase within 200 characters — Qwen is an LLM-based "
            "decoder and susceptible to repetition conditioning.",
            "The 0.6B model is faster but may drift from the reference voice on complex text; "
            "prefer shorter, simpler sentences.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Sentences over 200 characters",
            "Repeated phrases within the same chunk",
        ],
        "prompt_addendum": (
            "You are optimising text for Qwen Voice Clone (Qwen3-TTS Base model, 0.6B or "
            "1.7B), which clones a voice from reference audio using autoregressive LM "
            "generation, chunking input at 200-character boundaries.  "
            "Write sentences under 200 characters.  "
            "Match the rhythm and register of the reference audio if known.  "
            "Use ellipsis for pauses and em-dashes for pivots.  "
            "Spell out all numbers, abbreviations, and special characters.  "
            "Avoid repeating the same phrase within a single chunk — the LLM decoder can lock "
            "into repetition loops.  "
            "Do NOT emit SSML tags or bracket/parenthetical cues."
        ),
        "max_chunk_chars": 200,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 24000,
        "notes": (
            "Models: Qwen/Qwen3-TTS-12Hz-0.6B-Base and -1.7B-Base.  Supports chunking, "
            "conversation mode, and ebook mode.  Whisper used for reference transcription.  "
            "Multilingual (Chinese, English, Japanese, Korean, French, German, Spanish, "
            "Portuguese, Russian)."
        ),
    },
    # --------------------------------------------------------- Qwen Custom Voice
    "Qwen Custom Voice": {
        "engine_id": "qwen_custom_voice",
        "display_name": "🎭 Qwen Custom Voice",
        "script_rules": [
            "Qwen Custom Voice uses one of 9 preset speakers with optional style instructions "
            "— write plain narration text that the engine will render in the chosen speaker's "
            "voice.",
            "Keep sentences 50–200 characters; this mode does not chunk.",
            "Use ellipsis ('...') for pauses and em-dashes ('—') for breaks.",
            "Spell out all numbers, abbreviations, and special characters.",
            "Avoid SSML tags and bracket stage directions.",
            "Available speakers: Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, "
            "Uncle_fu, Vivian.  Match script register to the chosen speaker's typical "
            "use case (e.g. Serena for polished narration, Aiden for casual delivery).",
            "Style instructions are passed via the UI, not embedded in the script — keep the "
            "text free of style directives.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Style instructions embedded in the text (use the UI style parameter instead)",
            "Sentences over 250 characters",
        ],
        "prompt_addendum": (
            "You are optimising text for Qwen Custom Voice (Qwen3-TTS-CustomVoice model, 0.6B "
            "or 1.7B), which synthesises speech using one of 9 preset speakers (Aiden, Dylan, "
            "Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian).  "
            "Write clean narration text; style and speaker are set via the UI, not in the "
            "script.  "
            "Keep sentences 50–200 characters.  "
            "Use ellipsis for pauses and em-dashes for breaks.  "
            "Spell out all numbers, abbreviations, and special characters.  "
            "Do NOT embed style instructions in the text and do NOT emit SSML tags or "
            "bracket/parenthetical cues."
        ),
        "max_chunk_chars": 200,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": False,
        "sample_rate_hz": 24000,
        "notes": (
            "Models: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice and -1.7B-CustomVoice.  "
            "9 preset speakers.  Languages: Chinese, English, Japanese, Korean, French, "
            "German, Spanish, Portuguese, Russian.  No chunking in this mode."
        ),
    },
    # ----------------------------------------------------------------- VibeVoice
    "VibeVoice": {
        "engine_id": "vibevoice",
        "display_name": "🎸 VibeVoice",
        "script_rules": [
            "VibeVoice (1.5B diffusion-based model) creates expressive voice from reference "
            "audio — match script register closely to the reference.",
            "Sentences of 50–300 characters work well; the diffusion decoder handles "
            "moderate-length inputs gracefully.",
            "Use ellipsis ('...') for pauses and em-dashes ('—') for pivots.",
            "Spell out all numbers, abbreviations, and special characters.",
            "Avoid SSML tags and bracket stage directions.",
            "Flash attention provides a speed boost but does not change the output — "
            "no script changes needed based on flash attention state.",
            "Voice cloning quality depends on 5–15 seconds of clean reference audio; "
            "ensure the script's tone matches the reference.",
        ],
        "avoid": [
            "SSML tags",
            "Bracket or parenthetical stage directions",
            "Very short isolated single words without context",
        ],
        "prompt_addendum": (
            "You are optimising text for VibeVoice, a 1.5B diffusion-based TTS model that "
            "generates expressive speech conditioned on reference audio.  "
            "Write sentences 50–300 characters that match the register and tone of the "
            "reference audio.  "
            "Use ellipsis for deliberate pauses and em-dashes for dramatic breaks.  "
            "Spell out all numbers, abbreviations, and special characters.  "
            "Do NOT emit SSML tags or bracket/parenthetical cues.  "
            "The engine derives all expressiveness from the reference audio — clean, natural "
            "text is more effective than decorated text."
        ),
        "max_chunk_chars": 300,
        "supports_emotion_tags": False,
        "supports_ssml": False,
        "voice_cloning": True,
        "sample_rate_hz": 24000,
        "notes": (
            "Model: VibeVoice-1.5B (diffusion-based, 5 inference steps by default).  "
            "Flash attention option for faster inference.  "
            "Model directory scanned automatically from the 'models/' folder.  "
            "24 kHz output."
        ),
    },
}


def get_engine_prompt_addendum(engine_name: str) -> str:
    """Return the LLM system prompt addendum for the specified engine.

    The addendum is a concise block of instructions addressed to an LLM that is
    transforming raw text into TTS-ready narration.  Callers should append it to
    the base ``DEFAULT_LLM_NARRATION_SYSTEM_PROMPT`` before submitting to the LLM.

    Args:
        engine_name: The engine display name as used in the engine registry and UI
            dropdown (e.g. ``"Kokoro TTS"``, ``"IndexTTS2"``).

    Returns:
        Prompt addendum string, or empty string if the engine has no profile.
    """
    profile = ENGINE_SCRIPT_PROFILES.get(engine_name)
    if not profile:
        return ""
    return profile.get("prompt_addendum", "")


def get_engine_script_rules(engine_name: str) -> list[str]:
    """Return the list of atomic script optimisation rules for the specified engine.

    Each rule is a single, independently actionable guideline.  The list is
    suitable for display in documentation, agent skill files, or structured
    prompt injection.

    Args:
        engine_name: The engine display name as used in the engine registry and UI
            dropdown.

    Returns:
        List of rule strings, or empty list if the engine has no profile.
    """
    profile = ENGINE_SCRIPT_PROFILES.get(engine_name)
    if not profile:
        return []
    return list(profile.get("script_rules", []))


def get_engine_max_chunk_chars(engine_name: str) -> int:
    """Return the recommended maximum input characters per TTS chunk for an engine.

    Args:
        engine_name: The engine display name.

    Returns:
        Maximum character count, or 300 as a safe default if the engine has no profile.
    """
    profile = ENGINE_SCRIPT_PROFILES.get(engine_name)
    if not profile:
        return 300
    return int(profile.get("max_chunk_chars", 300))


def list_engines_with_profiles() -> list[str]:
    """Return a sorted list of all engine names that have optimisation profiles.

    Returns:
        Sorted list of engine name strings.
    """
    return sorted(ENGINE_SCRIPT_PROFILES.keys())
