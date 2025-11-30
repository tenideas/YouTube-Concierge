from __future__ import annotations

from typing import Dict

from domain import VideoCategory

# -- Classifier Prompts --

CLASSIFIER_SYSTEM_INSTRUCTION: str = (
    "You are a precise video content classifier. Your job is to assign each "
    "YouTube video transcript to exactly one category from a fixed list of "
    "identifiers."
)

_CLASSIFIER_ALLOWED_LABELS: str = ", ".join(
    category.name for category in VideoCategory
)

CLASSIFIER_PROMPT_TEMPLATE: str = (
    "You will be given metadata and a transcript from a YouTube video.\n"
    "Your task is to classify this video into exactly one category from the "
    "following list of identifiers:\n"
    f"{_CLASSIFIER_ALLOWED_LABELS}\n\n"
    "Video Metadata:\n"
    "- Title: {title}\n"
    "- Channel/Author: {author}\n\n"
    "Transcript (Snippet):\n"
    "---------------------\n"
    "{transcript_text}\n"
    "---------------------\n"
    "Rules:\n"
    "1. Respond with only one identifier.\n"
    "2. Do not include any explanation or extra text.\n"
    "3. The identifier must match exactly one of the listed values.\n\n"
    "Answer with only the identifier:"
)

# -- Summarizer Prompts --

GENERIC_SUMMARY_INSTRUCTION: str = (
    "Write a clear, concise summary of the video that captures the main "
    "ideas, key points, and any important conclusions. Assume the reader has "
    "not watched the video."
)

UNCERTAIN_SUMMARY_INSTRUCTION: str = (
    "You do not know the precise category of this video. Write a general "
    "summary that covers the main topics, structure, and any key takeaways in "
    "a way that would help someone decide whether to watch it."
)

SUMMARY_PROMPT_TEMPLATE: str = (
    "{instruction}\n"
    "{language_instruction}\n\n"
    "Transcript:\n"
    "---------------------\n"
    "{transcript_text}\n"
    "---------------------\n"
    "Summary:"
)

VIDEO_SUMMARY_INSTRUCTIONS: Dict[VideoCategory, str] = {
    VideoCategory.VLOG: (
        "Summarize this vlog by describing the main events, locations, and "
        "emotional beats of the video. Highlight what happens in roughly the "
        "order it occurs and what makes the video interesting to watch."
    ),
    VideoCategory.EDUCATIONAL_EXPLAINER: (
        "Summarize this educational explainer by clearly stating the main "
        "topic, the core concepts that are taught, and any step-by-step "
        "reasoning or examples used to explain them."
    ),
    VideoCategory.LECTURE_PRESENTATION: (
        "Summarize this lecture or presentation by outlining the main thesis, "
        "the key sections, and the most important arguments or results "
        "covered by the speaker."
    ),
    VideoCategory.DOCUMENTARY: (
        "Summarize this documentary-style video by describing the central "
        "subject, the narrative arc, and the most important facts, events, or "
        "interviews presented."
    ),
    VideoCategory.HISTORY_EXPLAINER: (
        "Summarize this history-related video by stating the historical "
        "period or events it covers, the main storyline, and the key causes, "
        "consequences, or insights discussed."
    ),
    VideoCategory.SCIENCE_NEWS: (
        "Summarize this science-focused news or update by stating the main "
        "scientific topic, what is new or important, and any implications or "
        "limitations mentioned."
    ),
    VideoCategory.ECONOMICS_EXPLAINER: (
        "Summarize this economics-related explainer by identifying the "
        "economic concept or situation, the main arguments or models used, "
        "and any key examples or policy implications."
    ),
    VideoCategory.NEWS_REPORT: (
        "Summarize this news report by clearly stating what happened, where "
        "and when it occurred, who is involved, and any known causes or next "
        "steps mentioned."
    ),
    VideoCategory.COMEDY_SKETCH: (
        "Summarize this comedy or sketch by describing the basic premise, the "
        "main characters, and the most important beats of the joke or story "
        "without trying to recreate the humor."
    ),
    VideoCategory.INTERVIEW_CONVERSATION: (
        "Summarize this interview or conversation by stating who is speaking, "
        "the main topics discussed, and any notable opinions, stories, or "
        "insights they share."
    ),
    VideoCategory.MUSIC_ANALYSIS: (
        "Summarize this music-related analysis by describing what music or "
        "artist is being discussed, the main musical or lyrical points the "
        "speaker makes, and any conclusions they draw."
    ),
    VideoCategory.TRAVEL_GUIDE: (
        "Summarize this travel-related video by listing the main places "
        "visited, notable experiences or tips, and any practical advice that "
        "would help someone planning a similar trip."
    ),
    VideoCategory.OTHER: GENERIC_SUMMARY_INSTRUCTION,
}

COMPARISON_SYSTEM_INSTRUCTION: str = (
    "You are an expert comparative analyst. Your job is to objectively compare "
    "multiple video sources based on specific user criteria."
)

COMPARISON_PROMPT_TEMPLATE: str = (
    "User Question/Criteria: {question}\n\n"
    "--- Source Materials ---\n"
    "{content_context}\n"
    "------------------------\n\n"
    "Instructions:\n"
    "1. Compare the videos specifically addressing the user's question.\n"
    "2. Cite specific examples or topics from the transcripts where possible.\n"
    "3. If the videos are unrelated to the question, state that clearly.\n\n"
    "Comparison:"
)
