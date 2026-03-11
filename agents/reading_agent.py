# agents/reading_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  IELTS Reading Agent
#
# Responsibilities:
#   1. Convert an IELTS reading PDF to paragraph images (A, B, C, …)
#      using PyMuPDF — no OCR, direct PDF-to-image rendering.
#   2. Extract comprehension questions as text for Q&A interaction.
#   3. Deliver paragraphs with ADHD-adaptive guided prompts.
#   4. Evaluate answers and generate adaptive hints.
#   5. Self-optimise the study strategy based on performance and
#      cognitive state, logging recommendations for the learner.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from core.llm_client import LLMClient


def generate_passage_prompt(passage_title: str, full_text: str, ld_profile: dict) -> str:
    """Generates a high-level reading prompt for the entire passage."""
    llm = LLMClient()
    
    adaptive_instructions = []
    all_ld = set(ld_profile.get('confirmed', []) + ld_profile.get('suspected', []))
    if 'adhd' in all_ld:
        adaptive_instructions.append(
            "The user has ADHD. Start with a ⚡ emoji. Give a very short, concrete instruction to help them start, like identifying the main subject. Keep it under 40 words."
        )
    else:
        adaptive_instructions.append(
            "Provide a brief, high-level reading strategy for the entire passage. Suggest what the user should pay attention to overall. Keep it concise."
        )

    prompt = f"""
    The user is about to read a passage titled "{passage_title}".
    Here is the full text:
    ---
    {full_text}
    ---
    Your task is to generate a single, high-level reading prompt to help the user focus before they begin reading paragraph by paragraph.

    [ADAPTIVE INSTRUCTIONS]
    {adaptive_instructions[0]}
    """

    try:
        guidance = llm.chat(
            system="You are a helpful reading coach. Your goal is to give a single, effective starting instruction for the entire passage.",
            user=prompt,
            temperature=0.5,
            max_tokens=100,
        )
        return guidance
    except Exception as e:
        return f"As you read, think about the main argument of the passage '{passage_title}'."


def generate_paragraph_guidance(paragraph_text: str, ld_profile: dict) -> str:
    """Generates AI-powered guidance for a single paragraph."""
    llm = LLMClient()
    
    adaptive_instructions = []
    all_ld = set(ld_profile.get('confirmed', []) + ld_profile.get('suspected', []))
    if 'adhd' in all_ld:
        adaptive_instructions.append(
            "The user has ADHD. Start with a ⚡ emoji. Ask a direct, engaging question to anchor their focus on this specific paragraph. Keep the guidance under 50 words."
        )
    else:
        adaptive_instructions.append(
            "Ask a thought-provoking question about this paragraph's main idea or purpose to encourage critical thinking. Keep it concise."
        )

    prompt = f"""
    The user is reading the following paragraph:
    ---
    {paragraph_text}
    ---
    Your task is to generate a brief, engaging reading prompt to guide them for this specific paragraph.

    [ADAPTIVE INSTRUCTIONS]
    {adaptive_instructions[0]}
    """

    try:
        guidance = llm.chat(
            system="You are a helpful reading coach. Your goal is to improve the user's reading strategy with a single, focused question for the given paragraph.",
            user=prompt,
            temperature=0.6,
            max_tokens=100,
        )
        return guidance
    except Exception as e:
        # Fallback for when LLM is not available
        return "What is the main idea of this paragraph?"


# ── Question page detection ───────────────────────────────────────
#
# IELTS reading PDFs contain two types of pages:
#   - Passage pages: contain the reading text (paragraphs A, B, C, …)
#   - Question pages: contain comprehension questions, always headed by a
#     marker such as "Questions 1–3", "Questions 10 and 11", etc.
#
# Robust detection is non-trivial because PDF text extraction can:
#   - Split "Questions" from "25–26" across lines/blocks
#   - Render en-dashes (–, U+2013) or em-dashes (—, U+2014) instead of hyphens
#   - Introduce unexpected whitespace or letter-case variation
#
# We therefore use a multi-signal approach rather than a single regex.

# Signal 1: explicit question-group header in any dash / "and" / "to" form.
# Supports: "Questions 1-3", "Questions 4–9", "Questions 10 and 11",
#           "Questions 1 to 3", "Question 1" (singular, no range).
_Q_HEADER_RE = re.compile(
    r'\bquestions?\s+\d+\s*(?:[-\u2013\u2014]\s*\d+|\band\b\s*\d+|\bto\b\s*\d+)?',
    re.IGNORECASE,
)

# Signal 2 (fallback): lines that start with a bare question number, e.g.:
#   "25 The track was originally built …"
#   "4   What does the author suggest …"
# We count how many such lines exist on the page.
_Q_NUMBERED_LINE_RE = re.compile(r'(?m)^\s*\d{1,3}\s+\S')

# Minimum number of numbered lines required for the fallback heuristic.
_Q_NUMBERED_LINE_THRESHOLD = 3


def detect_question_pages(doc) -> Set[int]:
    """Return the set of 0-based page indices that are IELTS question pages.

    Detection uses two signals:
      1. The page contains a "Questions m–n" / "Questions m and n" / "Question m"
         header in any dash variant (-, –, —).  This is the primary, high-precision
         signal and immediately classifies the page as a question page.
      2. The page contains at least ``_Q_NUMBERED_LINE_THRESHOLD`` lines that begin
         with a bare question number (e.g. "25 The track …").  This fallback catches
         question pages whose header was not extracted cleanly by PyMuPDF.

    Args:
        doc: An open ``fitz.Document`` (PyMuPDF).

    Returns:
        Set of page numbers (0-based) that are classified as question pages.
    """
    question_page_indices: Set[int] = set()
    for page_num, page in enumerate(doc):
        text = page.get_text()
        # Primary signal: explicit "Questions m…" header.
        if _Q_HEADER_RE.search(text):
            question_page_indices.add(page_num)
            continue
        # Fallback signal: multiple numbered question lines.
        if len(_Q_NUMBERED_LINE_RE.findall(text)) >= _Q_NUMBERED_LINE_THRESHOLD:
            question_page_indices.add(page_num)
    return question_page_indices


# ── PDF paragraph image extraction ───────────────────────────────

def extract_paragraph_images(
    pdf_bytes: bytes,
    output_dir: str,
    passage_id: int,
    zoom: float = 2.0,
) -> List[Dict[str, Any]]:
    """Convert an IELTS PDF to one high-resolution image per paragraph.

    Paragraphs are identified by their single-letter label (A–I or A–Z)
    that appears at the start of a text block on the passage pages.
    The questions pages are detected and skipped for image rendering,
    but their text is extracted separately via extract_questions_from_pdf().

    Args:
        pdf_bytes:   Raw bytes of the uploaded PDF file.
        output_dir:  Absolute filesystem path to save the PNG images.
        passage_id:  Used to namespace the output filenames.
        zoom:        Render scale (2.0 = 144 DPI).  Higher = sharper but larger.

    Returns:
        List of dicts:
        [
          {
            'label':      'A',         # paragraph letter
            'order':      1,           # 1-based index
            'image_path': 'passage_images/1_para_A.png',  # relative to MEDIA_ROOT
            'heading':    'Paragraph A',
            'body':       '',          # empty – passage is image-only
          },
          …
        ]
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is required for PDF-to-image conversion. "
            "Run: pip install pymupdf"
        )

    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(zoom, zoom)

    # ── Step 1: collect all text blocks with page + position info ──
    all_blocks: List[Dict] = []
    page_pixmaps: List[Any] = []
    page_rects: List[Any] = []

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        page_pixmaps.append(pix)
        page_rects.append(page.rect)

        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            all_blocks.append({
                'page': page_num,
                'x0': b[0], 'y0': b[1], 'x1': b[2], 'y1': b[3],
                'text': text,
            })

    # ── Step 2: find IELTS paragraph labels (A–Z at line/block start) ──
    # Typical IELTS paragraph: block starts with single capital letter followed
    # by a space/tab and then alphabetic content (not a question number).

    # Detect question pages using the robust multi-signal detector so we can
    # skip them entirely and avoid misidentifying question items as paragraph labels.
    question_page_indices = detect_question_pages(doc)

    para_blocks: List[Dict] = []
    para_label_re = re.compile(r'^([A-Z])\s+[A-Z\u2018\u201C"\'(]')

    for blk in all_blocks:
        # Skip all blocks on question pages entirely
        if blk['page'] in question_page_indices:
            continue
        # Skip blocks that look like question markers (e.g. "Questions 1-5")
        if re.match(r'^Questions?\s+\d', blk['text'], re.IGNORECASE):
            continue
        m = para_label_re.match(blk['text'])
        if m:
            para_blocks.append({**blk, 'label': m.group(1)})

    # ── Step 3: deduplicate labels (keep first occurrence) ──
    seen_labels: set = set()
    unique_para_blocks: List[Dict] = []
    for pb in para_blocks:
        if pb['label'] not in seen_labels:
            seen_labels.add(pb['label'])
            unique_para_blocks.append(pb)
    # Sort alphabetically by label so paragraphs always appear A, B, C, D…
    unique_para_blocks.sort(key=lambda b: b['label'])

    # ── Step 4: crop one image per paragraph ──
    paragraphs: List[Dict] = []
    from PIL import Image

    for i, para in enumerate(unique_para_blocks):
        page_num = para['page']
        page_rect = page_rects[page_num]
        pix = page_pixmaps[page_num]

        # Determine the bottom boundary: top of the next para on the same page
        # or the bottom of the page.
        if i + 1 < len(unique_para_blocks) and unique_para_blocks[i + 1]['page'] == page_num:
            y_end = unique_para_blocks[i + 1]['y0'] - 4  # 4pt gap
        else:
            y_end = page_rect.height

        # Convert PDF coordinates to pixel coordinates
        y_start_px = int(para['y0'] * zoom) - 4  # small top padding
        y_end_px   = int(y_end * zoom) + 4
        y_start_px = max(0, y_start_px)
        y_end_px   = min(pix.height, y_end_px)

        if y_end_px <= y_start_px:
            continue

        # Crop from pix (fitz.Pixmap) → PIL Image → save
        img_bytes = pix.tobytes("png")
        full_img = Image.open(io.BytesIO(img_bytes))
        cropped = full_img.crop((0, y_start_px, full_img.width, y_end_px))

        rel_path = f"passage_images/{passage_id}_para_{para['label']}.png"
        abs_path = os.path.join(output_dir, f"{passage_id}_para_{para['label']}.png")
        cropped.save(abs_path, "PNG", optimize=True)

        paragraphs.append({
            'label':      para['label'],
            'order':      ord(para['label']) - ord('A') + 1,
            'image_path': rel_path,
            'heading':    f"Paragraph {para['label']}",
            'body':       '',           # body is image-only
        })

    doc.close()

    # Fallback: if no labeled paragraphs found, convert each passage page to an image.
    # Question pages are excluded to avoid rendering question content as passage images.
    if not paragraphs:
        paragraphs = _fallback_page_images(
            pdf_bytes, output_dir, passage_id, zoom,
            question_page_indices=question_page_indices,
        )

    return paragraphs


def _fallback_page_images(
    pdf_bytes: bytes,
    output_dir: str,
    passage_id: int,
    zoom: float = 2.0,
    question_page_indices: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Fallback: one image per passage page when no paragraph labels are found.

    Question pages are excluded from the output so that comprehension-question
    content is not accidentally rendered as passage images.

    Args:
        pdf_bytes:             Raw bytes of the uploaded PDF.
        output_dir:            Absolute path to save PNG images.
        passage_id:            Used to namespace output filenames.
        zoom:                  Render scale factor.
        question_page_indices: Set of 0-based page indices to skip.  If *None*
                               the function falls back to detecting them itself.
    """
    try:
        import fitz
    except ImportError:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if question_page_indices is None:
        question_page_indices = detect_question_pages(doc)
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    passage_page_nums = [i for i in range(len(doc)) if i not in question_page_indices]
    for order, page_num in enumerate(passage_page_nums, start=1):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        rel_path = f"passage_images/{passage_id}_page_{order}.png"
        abs_path = os.path.join(output_dir, f"{passage_id}_page_{order}.png")
        pix.save(abs_path)
        pages.append({
            'label':      str(order),
            'order':      order,
            'image_path': rel_path,
            'heading':    f"Page {order}",
            'body':       '',
        })
    doc.close()
    return pages


def extract_questions_from_pdf(pdf_bytes: bytes) -> List[str]:
    """Extract only the comprehension questions text from the PDF.

    This reads the native PDF text structure (not OCR) and looks for
    numbered question items (1., 2., …) in the questions section.
    Only question pages are scanned; passage pages are ignored.
    """
    try:
        import fitz
    except ImportError:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    question_page_indices = detect_question_pages(doc)
    if question_page_indices:
        questions_raw = '\n'.join(doc[i].get_text() for i in sorted(question_page_indices))
    else:
        # No question pages detected; fall back to global split heuristic.
        raw = '\n'.join(page.get_text() for page in doc)
        _, questions_raw = _split_passage_and_questions(raw)
    doc.close()
    return _extract_questions(questions_raw)


def extract_question_groups_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract questions with their group labels from the PDF.

    Only question pages (detected via ``detect_question_pages``) contribute
    to the extracted text, preventing passage content from contaminating the
    question block.  Group headers of the form:

      "Questions 1-3"    (hyphen)
      "Questions 4–9"    (en-dash, U+2013)
      "Questions 10 and 11"
      "Questions 1 to 3"

    are all recognised and preserved as ``group_label`` on each question item.

    Returns a list of dicts:
        [
          {
            'order':             1,           # question number (1-based)
            'text':              'Choose…',   # question body text
            'group_label':       'Questions 1-3',  # enclosing group header (may be '')
            'group_instruction': 'Do the following…',  # full group instruction (may be '')
          },
          …
        ]
    """
    try:
        import fitz
    except ImportError:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    question_page_indices = detect_question_pages(doc)
    # Extract full-document text upfront for fallback use
    full_doc_text = '\n'.join(page.get_text() for page in doc)
    if question_page_indices:
        # Use only the text from detected question pages for maximum precision.
        questions_raw = '\n'.join(doc[i].get_text() for i in sorted(question_page_indices))
    else:
        # No question pages detected; fall back to global split heuristic.
        _, questions_raw = _split_passage_and_questions(full_doc_text)
    doc.close()

    results = _extract_question_groups(questions_raw)

    # Fallback 1: if primary extraction returned nothing and we had question pages,
    # try scanning the full document text (question pages detection may have missed some).
    if not results and full_doc_text and full_doc_text != questions_raw:
        results = _extract_question_groups(full_doc_text)

    # Fallback 2: universal extractor on the questions section text
    if not results:
        results = _universal_question_extractor(questions_raw)

    # Fallback 3: universal extractor on the entire document text
    if not results and full_doc_text:
        results = _universal_question_extractor(full_doc_text)

    return results


# ── Text helpers (for question extraction only) ───────────────────

def _split_passage_and_questions(raw: str) -> Tuple[str, str]:
    """Return (passage_text, questions_raw) by heuristic splitting.

    Strategy: find the LAST occurrence of a question-block header so that
    any accidental match inside the passage body is skipped.  The content
    before the header is always treated as the passage; the content from the
    header onward is always treated as the questions block.

    The old '< 20 % swap' heuristic is intentionally removed: it caused the
    passage and questions to be silently transposed when a question-like
    phrase appeared early in the passage text.
    """
    # Try to find a question-block header. Use finditer and take the LAST match
    # so that any accidental occurrence inside the passage body is ignored.
    q_marker_re = re.compile(
        r'^(Questions?\s+\d|Question\s+\d|\*\*Questions|'
        r'QUESTIONS|Questions and answers|Reading comprehension questions'
        r'|READING COMPREHENSION|Comprehension Questions)',
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(q_marker_re.finditer(raw))
    if matches:
        # Use the LAST match – question headers appear near the end of the PDF
        q_marker = matches[-1]
        before = raw[: q_marker.start()].strip()
        after  = raw[q_marker.start():].strip()
        return before, after

    # Fallback: look for the first numbered item "1. " or "1) "
    first_q = re.search(r'^\s*1[\.\)]\s+\S', raw, re.MULTILINE)
    if first_q:
        before = raw[: first_q.start()].strip()
        after  = raw[first_q.start():].strip()
        return before, after

    # No question section found at all
    return raw.strip(), ''


def _extract_questions(questions_raw: str) -> List[str]:
    """Parse individual question strings from the questions block."""
    if not questions_raw:
        return []

    parts = re.split(r'(?=^\s*\d+[\.\)]\s)', questions_raw, flags=re.MULTILINE)
    questions = []
    for part in parts:
        cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', part.strip())
        if re.match(r'^Questions?\s+[\d–\-]+', cleaned, re.IGNORECASE):
            continue
        if cleaned:
            questions.append(cleaned.strip())
    return questions


# Regex matching IELTS question group headers.  Supports all real-world forms:
#   "Questions 1-3"        (ASCII hyphen)
#   "Questions 4–9"        (en-dash U+2013)
#   "Questions 25—26"      (em-dash U+2014)
#   "Questions 10 and 11"
#   "Questions 1 to 3"
#   "Question 1"           (singular, no range)
_GROUP_HEADER_RE = re.compile(
    r'^(Questions?\s+\d+(?:\s*[-\u2013\u2014]\s*\d+|\s+and\s+\d+|\s+to\s+\d+)?[^\n]*)',
    re.IGNORECASE | re.MULTILINE,
)


def _extract_question_groups(questions_raw: str) -> List[Dict[str, Any]]:
    """Parse questions from the questions block, preserving their group labels.

    Returns a list of dicts:
        [{'order': int, 'text': str, 'group_label': str, 'group_instruction': str}, …]
    """
    if not questions_raw:
        return []

    results: List[Dict[str, Any]] = []
    current_group = ''
    current_group_instruction = ''

    # Split the block on either a group header OR a numbered question start.
    # The group-header pattern is kept in sync with _GROUP_HEADER_RE above and
    # also matches bare numbered question lines (e.g. "25 The track …") used on
    # IELTS question pages that omit the trailing "." or ")".
    token_re = re.compile(
        r'(?=^Questions?\s+\d+(?:\s*[-\u2013\u2014]\s*\d+|\s+and\s+\d+|\s+to\s+\d+)?)'
        r'|(?=^\s*\d+[\.\)]\s)'
        r'|(?=^\s*\d+\s+\S)',
        re.IGNORECASE | re.MULTILINE,
    )
    segments = token_re.split(questions_raw)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Check if this segment is a group header
        if _GROUP_HEADER_RE.match(seg):
            lines = seg.split('\n')
            # First line is the group label
            current_group = lines[0].strip()
            # Lines after the header (before any numbered question) are the instruction
            instr_lines = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                # Stop if we hit a numbered question line
                if re.match(r'^\d+[\.\)]\s', line) or re.match(r'^\d+\s+\S', line):
                    break
                instr_lines.append(line)
            current_group_instruction = ' '.join(instr_lines).strip()
            continue

        # Check if this segment is a numbered question item (with "." / ")")
        q_match = re.match(r'^\s*(\d+)[\.\)]\s*(\S.*)', seg, re.DOTALL)
        if not q_match:
            # Also accept bare numbered lines: "25 The track was originally …"
            q_match = re.match(r'^\s*(\d+)\s+(\S.*)', seg, re.DOTALL)
        if q_match:
            q_num  = int(q_match.group(1))
            q_text = re.sub(r'\s+', ' ', q_match.group(2)).strip()
            if q_text:
                results.append({
                    'order':             q_num,
                    'text':              q_text,
                    'group_label':       current_group,
                    'group_instruction': current_group_instruction,
                })

    # If no grouped questions were found, fall back to the simple extractor
    if not results:
        for i, text in enumerate(_extract_questions(questions_raw), start=1):
            results.append({
                'order':             i,
                'text':              text,
                'group_label':       '',
                'group_instruction': '',
            })

    return results


def _universal_question_extractor(text: str) -> List[Dict[str, Any]]:
    """Universal fallback question extractor — tries multiple pattern strategies.

    Used as a last resort when both primary extraction and the standard fallback
    return 0 results.  Scans line-by-line for bare numbered items.

    Returns a list of dicts compatible with _extract_question_groups output.
    """
    if not text:
        return []

    results: List[Dict[str, Any]] = []
    lines = text.split('\n')
    current_q_num: Optional[int] = None
    current_q_lines: List[str] = []

    def _flush_question() -> None:
        if current_q_num is not None and current_q_lines:
            q_text = re.sub(r'\s+', ' ', ' '.join(current_q_lines)).strip()
            if q_text:
                results.append({
                    'order':             current_q_num,
                    'text':              q_text,
                    'group_label':       '',
                    'group_instruction': '',
                })

    for line in lines:
        line_stripped = line.strip()

        # Match: "1. Text…", "1) Text…", "1 Text starting with any character"
        m = re.match(r'^(\d{1,3})[\.\)]\s+(.+)', line_stripped)
        if not m:
            m = re.match(r'^(\d{1,3})\s+([^\d\s].+)', line_stripped)

        if m:
            _flush_question()
            current_q_num = int(m.group(1))
            current_q_lines = [m.group(2).strip()]
        elif current_q_num is not None and line_stripped:
            # Stop accumulating at a new section header or blank-then-capital pattern
            if re.match(r'^Questions?\s+\d', line_stripped, re.IGNORECASE):
                _flush_question()
                current_q_num = None
                current_q_lines = []
            else:
                current_q_lines.append(line_stripped)

    _flush_question()
    return results


# ── Guidance generation (LLM-free fallback) ──────────────────────

# Threshold: hints_used > questions_answered × HINT_USAGE_THRESHOLD → high hint usage warning
_HINT_USAGE_THRESHOLD = 1.5

# Regex for extracting alpha-only keywords of at least 3 characters
_KEYWORD_RE = re.compile(r'\b[A-Za-z]{3,}\b')

_ADHD_TIPS = [
    "⚡ Focus: look at the paragraph letter first — it anchors your place.",
    "⚡ Read the first sentence slowly, then let your eyes scan the rest.",
    "⚡ Underline names, dates, or numbers as you read.",
    "⚡ Take one sentence at a time. Pause and breathe after each one.",
    "⚡ If your mind wanders, return to the paragraph letter and restart.",
]

_GENERAL_TIPS = [
    "Skim the paragraph once quickly, then read it carefully.",
    "Pay attention to the topic sentence (usually the first sentence).",
    "Look for keywords that match the question wording.",
    "Note any signal words: 'however', 'therefore', 'in contrast'.",
    "Connect the paragraph to the passage title for context.",
]


def _build_section_intro(
    section: dict,
    section_num: int,
    total_sections: int,
    ld_profile: dict,
    attempt_score: Optional[float],
) -> str:
    """Generate a short guiding message to display before a paragraph."""
    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )

    lines = []
    lines.append(
        f"📖 Paragraph {section_num} of {total_sections}: **{section['heading']}**"
    )

    if attempt_score is not None and attempt_score < 0.5:
        lines.append(
            "\n💡 The previous paragraph was tricky — take your time here."
        )

    if 'adhd' in all_ld:
        tip = _ADHD_TIPS[(section_num - 1) % len(_ADHD_TIPS)]
        lines.append('\n' + tip)
    else:
        tip = _GENERAL_TIPS[(section_num - 1) % len(_GENERAL_TIPS)]
        lines.append('\n' + tip)

    lines.append("\nRead the paragraph image, then answer the questions on the right.")
    return '\n'.join(lines)


def _build_hint(
    question_text: str,
    section_body: str,
    ld_profile: dict,
    hints_used: int,
) -> str:
    """Generate a hint for the given question, escalating with more hints_used."""
    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )

    # Extract a candidate key sentence from the section
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', section_body) if len(s.strip()) > 30]

    # Find sentence most similar to question by naive keyword overlap
    q_words = set(re.findall(r'\b\w{4,}\b', question_text.lower()))
    best = max(sentences, key=lambda s: len(q_words & set(re.findall(r'\b\w{4,}\b', s.lower()))), default='')

    hint_lines = []

    if 'adhd' in all_ld:
        hint_lines.append("⚡ Focus: re-read the section carefully.")

    if hints_used == 0:
        hint_lines.append("💡 **Hint 1**: Look for keywords from the question in the text.")
    elif hints_used == 1:
        q_kw = [w for w in q_words if len(w) > 4][:4]
        if q_kw:
            hint_lines.append(
                f"💡 **Hint 2**: Try searching for: {', '.join(q_kw)}"
            )
        else:
            hint_lines.append("💡 **Hint 2**: Re-read the section and think about the main idea.")
    else:
        if best:
            preview = best[:120] + ('…' if len(best) > 120 else '')
            hint_lines.append(
                f"💡 **Hint 3**: The answer is near this sentence:\n\n> _{preview}_"
            )
        else:
            hint_lines.append("💡 **Hint 3**: Try re-reading the entire section one more time.")

    return '\n'.join(hint_lines)


def _evaluate_answer(
    user_answer: str,
    question_text: str,
    section_body: str,
) -> dict:
    """Simple heuristic answer evaluation.

    Returns {'correct': bool, 'score': float, 'feedback': str}
    """
    if not user_answer.strip():
        return {'correct': False, 'score': 0.0, 'feedback': 'Please write an answer before submitting.'}

    # Keyword overlap between answer and section
    answer_words = set(re.findall(r'\b\w{4,}\b', user_answer.lower()))
    section_words = set(re.findall(r'\b\w{4,}\b', section_body.lower()))
    q_words = set(re.findall(r'\b\w{4,}\b', question_text.lower()))

    # The answer should reference content from the section
    overlap_section = len(answer_words & section_words)
    overlap_question = len(answer_words & q_words)

    score = min(1.0, (overlap_section * 0.06 + overlap_question * 0.15))

    if score >= 0.5:
        feedback = "✅ Good answer! Your response uses relevant content from the passage."
    elif score >= 0.25:
        feedback = (
            "🟡 Partially correct. Try to include more details directly from the passage."
        )
    else:
        feedback = (
            "❌ Your answer may be off-track.  Re-read the section and look for specific details."
        )

    return {'correct': score >= 0.5, 'score': round(score, 2), 'feedback': feedback}


def _build_strategy(attempt_data: dict, ld_profile: dict) -> str:
    """Generate a personalised, adaptive study strategy based on attempt history.

    The strategy is informed by:
    • Score accuracy (proportion of correct answers)
    • Hint usage (frequency and distribution across questions)
    • Learner LD profile (ADHD, anxiety, etc.)

    Returns a Markdown-formatted strategy string.
    """
    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )
    answers = attempt_data.get('answers', {})
    hints_used = attempt_data.get('hints_used', 0)

    total = len(answers)
    correct = sum(1 for v in answers.values() if isinstance(v, dict) and v.get('correct'))
    pct = int(correct / total * 100) if total > 0 else 0

    lines = ['### 📊 Adaptive Learning Strategy\n']

    if total > 0:
        lines.append(f"**Your score:** {correct}/{total} ({pct}%)\n")

    # ── Performance-based strategy ───────────────────────────────
    if pct >= 80:
        lines.append(
            "✅ **Excellent work!** You have a strong grasp of the passage.\n\n"
            "💡 **Next-level tip:** Practice **inferential reading** — look for\n"
            "what the author *implies* rather than what they *state directly*.\n"
        )
    elif pct >= 50:
        lines.append(
            "🟡 **Good effort.** You understood the main ideas but missed some details.\n\n"
            "💡 **Strategy:** Use the **Scan-Locate-Verify** technique:\n"
            "1. **Scan** – read the question and circle key nouns/verbs\n"
            "2. **Locate** – find those exact words (or synonyms) in the passage\n"
            "3. **Verify** – re-read the surrounding sentences before answering\n"
        )
    else:
        lines.append(
            "❌ **Needs more practice.** Try reading each paragraph twice:\n"
            "once for gist, once for detail.\n\n"
            "💡 **SQ3R Method:**\n"
            "1. **Survey** – skim headings and bold words (30 seconds)\n"
            "2. **Question** – turn each paragraph label into a question\n"
            "3. **Read** – read carefully to answer your question\n"
            "4. **Recite** – close the passage and recall the main point\n"
            "5. **Review** – check your recall and correct any errors\n"
        )

    # ── Hint-based strategy ───────────────────────────────────────
    if hints_used > total * _HINT_USAGE_THRESHOLD and total > 0:
        lines.append(
            "\n⚠️ **Hint usage was high.** Before requesting a hint next time:\n"
            "- Re-read the paragraph once more at a slower pace\n"
            "- Highlight or underline every proper noun, number, and date\n"
            "- Ask yourself: 'Which sentence answers the question directly?'\n"
        )

    # ── ADHD-specific adaptive strategies ────────────────────────
    if 'adhd' in all_ld:
        lines.append("\n---\n### ⚡ ADHD Support Strategies\n")

        if pct < 50:
            lines.append(
                "**Chunking:** The passage is already split into labelled paragraphs.\n"
                "Treat each paragraph as a *separate mini-reading task*.\n"
                "Finish paragraph A completely before moving to B.\n\n"
            )

        lines.append(
            "**Body-doubling tip:** Read the passage aloud to yourself —\n"
            "hearing the words activates an extra attention channel.\n\n"
        )
        lines.append(
            "**Movement breaks:** After every 2 paragraphs, stand up, stretch,\n"
            "or do 10 jumping jacks. Physical movement resets focus.\n\n"
        )
        lines.append(
            "**Finger-tracking:** Use your finger or a ruler under each line\n"
            "to prevent eye-skipping and keep your place.\n\n"
        )
        if hints_used > 2:
            lines.append(
                "**External working memory:** Write a 1-sentence summary of\n"
                "each paragraph on paper *before* answering questions.\n"
                "This offloads memory demand and reduces the need for hints.\n"
            )

    # ── Anxiety-specific strategies ───────────────────────────────
    if 'anxiety' in all_ld:
        lines.append("\n---\n### 🌀 Anxiety Management Tips\n")
        lines.append(
            "Remember: every answer is *in the passage*. You don't need\n"
            "prior knowledge — only careful reading.\n\n"
        )
        if pct < 60:
            lines.append(
                "If you feel stuck, try the **'park and return'** strategy:\n"
                "skip the difficult question, answer easier ones first,\n"
                "then return with a clearer mind.\n"
            )

    return '\n'.join(lines)


# ── Floating assistant tip (proactive, performance-aware) ─────────

def _extract_heading_keywords(heading: str) -> List[str]:
    """Extract meaningful keywords from a section heading."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'it',
        'paragraph', 'section', 'part', 'this', 'that',
    }
    words = re.findall(r'\b[A-Za-z]{4,}\b', heading)
    return [w for w in words if w.lower() not in stop_words][:4]


def _build_assistant_tip(
    para_order: int,
    total_sections: int,
    answers: dict,
    hints_used: int,
    ld_profile: dict,
    current_section_heading: str = '',
    section_body: str = '',
    mode: str = 'auto',
) -> str:
    """Generate a proactive assistant tip based on current reading progress.

    Provides adaptive guidance including:
    • Encouragement and score feedback
    • Keyword extraction from the current paragraph heading / body
    • ADHD-specific focus tips
    • Hint-usage advice
    • Mode-specific adjustments (focus / calm / speed / auto)

    Args:
        mode: Assistant mode — ``"auto"`` (default, uses LD profile),
              ``"focus"`` (ADHD-optimised), ``"calm"`` (anxiety-optimised),
              or ``"speed"`` (minimal scaffolding).
    """
    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )

    # ── Resolve effective mode ────────────────────────────────────
    effective_mode = mode
    if effective_mode == 'auto':
        if 'adhd' in all_ld:
            effective_mode = 'focus'
        elif 'anxiety' in all_ld:
            effective_mode = 'calm'
        else:
            effective_mode = 'speed'  # treat as standard/direct for general learners

    total_answered = len(answers)
    correct = sum(1 for v in answers.values() if isinstance(v, dict) and v.get('correct'))
    pct = int(correct / total_answered * 100) if total_answered > 0 else None

    lines: List[str] = []

    # ── Position indicator ────────────────────────────────────────
    remaining = total_sections - para_order
    if remaining > 0:
        lines.append(
            f"📍 **Paragraph {para_order} of {total_sections}** — "
            f"{remaining} more to go."
        )
    else:
        lines.append(
            f"📍 **Paragraph {para_order} of {total_sections}** — "
            "this is the **last paragraph**!"
        )

    # ── Performance feedback ──────────────────────────────────────
    if pct is not None:
        if pct >= 70:
            lines.append(
                f"\n✅ You're scoring **{pct}%** — excellent reading comprehension!"
            )
        elif pct >= 40:
            lines.append(
                f"\n🟡 Current score: **{pct}%**. "
                "Try scanning for keywords that mirror the question wording."
            )
        else:
            lines.append(
                f"\n💪 Score so far: **{pct}%**. "
                "Every paragraph is a fresh chance — keep going!"
            )

    # ── Hint-usage advice ─────────────────────────────────────────
    if total_answered > 0 and hints_used > total_answered * _HINT_USAGE_THRESHOLD:
        lines.append(
            "\n⚠️ You've used quite a few hints. "
            "Before clicking **Hint**, try re-reading the first sentence "
            "of the paragraph — the topic sentence often contains the answer."
        )

    # ── Keyword extraction ────────────────────────────────────────
    # Combine heading + body to find domain-specific terms.
    keyword_source = (current_section_heading + ' ' + section_body).strip()
    if keyword_source:
        kw_candidates = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b', keyword_source)
        stop_words = {
            'about', 'above', 'after', 'again', 'against', 'along', 'also',
            'although', 'another', 'because', 'before', 'being', 'between',
            'during', 'every', 'first', 'found', 'great', 'however', 'include',
            'into', 'just', 'known', 'like', 'made', 'make', 'many', 'more',
            'most', 'much', 'must', 'never', 'often', 'only', 'other', 'over',
            'paragraph', 'people', 'section', 'since', 'some', 'such', 'than',
            'that', 'their', 'them', 'then', 'there', 'these', 'they', 'this',
            'those', 'through', 'time', 'under', 'until', 'upon', 'used',
            'very', 'when', 'where', 'which', 'while', 'with', 'within',
            'would', 'your',
        }
        seen: set = set()
        keywords: List[str] = []
        for w in kw_candidates:
            lw = w.lower()
            if lw not in stop_words and lw not in seen and len(lw) >= 5:
                seen.add(lw)
                keywords.append(w)
            if len(keywords) >= 5:
                break

        if keywords:
            lines.append(
                "\n🔑 **Key words to look for:** "
                + ', '.join(f'*{k}*' for k in keywords)
            )

    # ── ADHD-specific tip ────────────────────────────────────────
    ld_focus_lines: List[str] = []
    if 'adhd' in all_ld or effective_mode == 'focus':
        adhd_tips = [
            "⚡ Focus: trace each line with your finger as you read.",
            "⚡ Read the first sentence, pause, then continue.",
            "⚡ If you lose focus, come back to the paragraph heading and start again.",
            "⚡ One sentence at a time — don't rush.",
            "⚡ Glance at the paragraph heading before you start reading the body text.",
        ]
        tip = adhd_tips[(para_order - 1) % len(adhd_tips)]
        ld_focus_lines.append(f"\n{tip}")

    # ── Anxiety tip ──────────────────────────────────────────────
    ld_calm_lines: List[str] = []
    if 'anxiety' in all_ld or effective_mode == 'calm':
        ld_calm_lines.append(
            "\n🌀 Remember: every answer is *in the passage*. "
            "Trust your reading — you don't need prior knowledge."
        )

    # ── Mode-specific injections ──────────────────────────────────
    # Speed mode: skip LD-specific scaffolding and give direct guidance.
    if effective_mode == 'speed':
        lines.append(
            "\n🚀 **Speed Mode:** Scan for keywords from the question, "
            "locate the matching sentence in the paragraph, verify your answer."
        )
    elif effective_mode == 'focus':
        lines.extend(ld_focus_lines)
        # Shorter chunks + break reminders for ADHD/Focus mode
        if para_order % 2 == 0:
            lines.append(
                "\n⚡ **Focus Mode:** You've reached paragraph "
                f"{para_order}. Take a 30-second micro-break if needed — "
                "stand up, stretch, then return."
            )
        else:
            lines.append(
                "\n⚡ **Focus Mode:** Read *one sentence at a time*. "
                "Pause after each sentence and ask: \"What did that say?\""
            )
    elif effective_mode == 'calm':
        lines.extend(ld_calm_lines)
        lines.append(
            "\n🌀 **Calm Mode:** There is no time pressure here. "
            "Read at your own comfortable pace. "
            "If a question feels hard, park it and come back later."
        )
    else:
        # Default: include both LD-specific tips when applicable
        lines.extend(ld_focus_lines)
        lines.extend(ld_calm_lines)

    return '\n'.join(lines)


# ── Main agent entry point ────────────────────────────────────────

def reading_agent_guide_section(
    section: dict,
    section_num: int,
    total_sections: int,
    ld_profile: dict,
    attempt_score: Optional[float] = None,
) -> str:
    """Return the intro guidance text for displaying a new section."""
    return _build_section_intro(section, section_num, total_sections, ld_profile, attempt_score)


def reading_agent_hint(
    question_text: str,
    section_body: str,
    ld_profile: dict,
    hints_used: int,
) -> str:
    """Return a hint for the current question."""
    return _build_hint(question_text, section_body, ld_profile, hints_used)


def reading_agent_evaluate(
    user_answer: str,
    question_text: str,
    section_body: str,
) -> dict:
    """Evaluate a learner's answer."""
    return _evaluate_answer(user_answer, question_text, section_body)


def reading_agent_strategy(attempt_data: dict, ld_profile: dict) -> str:
    """Generate a personalised learning strategy."""
    return _build_strategy(attempt_data, ld_profile)


def reading_agent_assistant_tip(
    para_order: int,
    total_sections: int,
    answers: dict,
    hints_used: int,
    ld_profile: dict,
    current_section_heading: str = '',
    section_body: str = '',
    mode: str = 'auto',
) -> str:
    """Return a proactive assistant tip for the current reading state.

    Args:
        para_order:              Current paragraph number (1-based).
        total_sections:          Total paragraphs in the passage.
        answers:                 Dict mapping question IDs to answer result dicts.
        hints_used:              Total hint requests so far.
        ld_profile:              Learner LD profile dict.
        current_section_heading: Heading of the paragraph being viewed.
        section_body:            Body text of the current paragraph (may be empty
                                 when the section is image-only; raw_text is used
                                 as fallback in the view layer).
        mode:                    Assistant mode — ``"auto"`` (uses LD profile),
                                 ``"focus"`` (ADHD-optimised shorter chunks),
                                 ``"calm"`` (reassuring, no time pressure),
                                 or ``"speed"`` (minimal scaffolding).
    """
    return _build_assistant_tip(
        para_order=para_order,
        total_sections=total_sections,
        answers=answers,
        hints_used=hints_used,
        ld_profile=ld_profile,
        current_section_heading=current_section_heading,
        section_body=section_body,
        mode=mode,
    )


# ── Question-to-paragraph mapper ──────────────────────────────────

def _score_question_paragraph_overlap(question_text: str, para_body: str) -> float:
    """Score keyword overlap between a question and a paragraph body."""
    if not para_body:
        return 0.0
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'it', 'this', 'that',
        'these', 'those', 'which', 'who', 'what', 'where', 'when', 'how',
        'according', 'passage', 'paragraph', 'text', 'author', 'writer',
        'following', 'correct', 'true', 'false', 'not', 'given', 'question',
        'answer', 'statement',
    }

    def extract_keywords(text: str) -> set:
        words = _KEYWORD_RE.findall(text.lower())
        return {w for w in words if w not in stop_words}

    q_words = extract_keywords(question_text)
    p_words = extract_keywords(para_body)

    if not q_words:
        return 0.0
    overlap = len(q_words & p_words)
    return overlap / len(q_words)


def _distribute_evenly(
    qs: List[Dict],
    sections: List[Dict],
    mapping: Dict[int, List[int]],
) -> None:
    """Append question IDs to *mapping* using round-robin even distribution.

    Modifies *mapping* in-place; never replaces existing entries.

    The last section receives any surplus questions when
    ``len(qs) % len(sections) != 0``.
    """
    n = len(sections)
    if not n or not qs:
        return
    q_per_section = max(1, len(qs) // n)
    for i, q in enumerate(qs):
        idx = min(i // q_per_section, n - 1)
        mapping[sections[idx]['id']].append(q['id'])


def map_questions_to_paragraphs(
    sections: List[Dict],
    questions: List[Dict],
) -> Dict[int, List[int]]:
    """Map question IDs to section IDs based on content overlap or group labels.

    Strategy priority for **image-only** passages (no body text):
    1. **Group-label distribution** — questions that carry a non-empty
       ``group_label`` (e.g. "Questions 1-3") are kept together as a group and
       groups are distributed proportionally across sections.  Questions without
       a group label fall through to Strategy 3.
    2. (skipped — no body text available)
    3. **Even distribution** — remaining questions spread evenly.

    Strategy priority for **text** passages (sections have body text):
    2. **Keyword-overlap scoring** — each question is matched to the section
       whose body text shares the most keywords with the question.
    3. **Even distribution** — questions with no keyword overlap are spread
       evenly across sections.

    Args:
        sections:  List of dicts with keys: id, order, heading, body
        questions: List of dicts with keys: id, order, text,
                   and optionally group_label

    Returns:
        Dict mapping section_id -> list of associated question IDs.
    """
    if not sections or not questions:
        return {}

    n_sections = len(sections)
    mapping: Dict[int, List[int]] = {s['id']: [] for s in sections}

    has_body_text = any(s.get('body', '').strip() for s in sections)

    # ── Strategy 1: group-label–based distribution (image-only passages) ─
    has_groups = any(q.get('group_label', '') for q in questions)

    if has_groups and not has_body_text:
        grouped_qs: List[Dict] = []
        ungrouped_qs: List[Dict] = []
        for q in sorted(questions, key=lambda x: x.get('order', 0)):
            if q.get('group_label', ''):
                grouped_qs.append(q)
            else:
                ungrouped_qs.append(q)

        # Collect unique groups in the order they first appear
        seen_groups: Dict[str, List[Dict]] = {}
        for q in grouped_qs:
            seen_groups.setdefault(q['group_label'], []).append(q)

        groups = list(seen_groups.values())
        n_groups = len(groups)

        if n_groups:
            for g_idx, group_qs in enumerate(groups):
                sec_idx = min(int(g_idx * n_sections / n_groups), n_sections - 1)
                for q in group_qs:
                    mapping[sections[sec_idx]['id']].append(q['id'])

        # Ungrouped questions distributed evenly (Strategy 3)
        _distribute_evenly(ungrouped_qs, sections, mapping)
        return mapping

    # ── Strategy 2: keyword-overlap scoring (text passages) ──────
    unmapped: List[Dict] = []

    if has_body_text:
        for q in questions:
            scores = [
                (s['id'], _score_question_paragraph_overlap(q.get('text', ''), s.get('body', '')))
                for s in sections
            ]
            best_id, best_score = max(scores, key=lambda x: x[1])
            if best_score > 0.0:
                mapping[best_id].append(q['id'])
            else:
                unmapped.append(q)
    else:
        unmapped = list(questions)

    # ── Strategy 3: even distribution fallback ────────────────────
    _distribute_evenly(unmapped, sections, mapping)
    return mapping


def _build_paragraph_strategy(
    section: Dict,
    related_questions: List[Dict],
    ld_profile: dict,
) -> str:
    """Generate an inline reading strategy card for a paragraph and its questions.

    Returns a Markdown-formatted string, or an empty string if there are
    no related questions for this paragraph.
    """
    if not related_questions:
        return ''

    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )

    q_nums = ', '.join(
        f'Q{q["order"]}' for q in sorted(related_questions, key=lambda x: x['order'])
    )

    lines: List[str] = [f"**📌 Reading Strategy — {q_nums}**\n"]

    q_texts = ' '.join(q.get('text', '') for q in related_questions).lower()

    if any(w in q_texts for w in ['true', 'false', 'not given', 'yes', 'no']):
        lines.append(
            f"These questions test **factual accuracy**. "
            f"As you read this paragraph, check whether each statement in "
            f"{q_nums} *agrees with*, *contradicts*, or is *not mentioned* in the text."
        )
    elif any(w in q_texts for w in ['heading', 'title', 'main idea', 'which paragraph']):
        lines.append(
            f"These questions test **main idea comprehension**. "
            f"Focus on the *first and last sentence* of this paragraph — "
            f"they usually contain the paragraph's central point."
        )
    elif any(w in q_texts for w in ['according', 'what', 'when', 'where', 'how many', 'how much', 'who']):
        key_terms: List[str] = []
        for q in related_questions:
            words = re.findall(r'\b[A-Z][a-z]{3,}\b', q.get('text', ''))
            key_terms.extend(words[:2])
        key_terms = list(dict.fromkeys(key_terms))[:4]
        focus = f" Pay attention to: **{', '.join(key_terms)}**." if key_terms else ''
        lines.append(
            f"These questions test **specific details**.{focus} "
            f"Scan for exact words, numbers, dates, and names that match the questions."
        )
    else:
        lines.append(
            f"Read this paragraph carefully to answer {q_nums}. "
            f"Look for keywords from the questions that appear in the text."
        )

    if 'adhd' in all_ld:
        lines.append(
            f"\n⚡ *ADHD tip: Before reading, quickly glance at {q_nums} "
            f"to prime your attention on what to look for.*"
        )

    return '\n'.join(lines)


def reading_agent_paragraph_strategy(
    section: Dict,
    related_questions: List[Dict],
    ld_profile: dict,
) -> str:
    """Return an inline reading strategy for a specific paragraph.

    Args:
        section:           Dict with keys id, order, heading, body.
        related_questions: List of question dicts ({id, order, text}) mapped to
                           this paragraph by map_questions_to_paragraphs().
        ld_profile:        Learner LD profile dict.

    Returns:
        Markdown-formatted strategy string, or '' if no questions apply.
    """
    return _build_paragraph_strategy(section, related_questions, ld_profile)


# ── Preflight Tips: AI pre-solves questions → reading tips ────────

def _infer_tips_for_section(
    section_order: int,
    related_questions: List[Dict],
    ld_profile: dict,
) -> List[str]:
    """Generate pre-reading tips for a section based on its questions.

    Simulates what an expert reader would notice when pre-reading the
    questions before reading the passage.  Returns a list of tip strings
    (plain text, no Markdown).
    """
    if not related_questions:
        return []

    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )
    tips: List[str] = []
    q_texts_combined = ' '.join(q.get('text', '') for q in related_questions).lower()
    q_nums = ', '.join(
        f'Q{q["order"]}' for q in sorted(related_questions, key=lambda x: x['order'])
    )

    # ── Tip 1: Question-type tip ──────────────────────────────────
    if any(w in q_texts_combined for w in ['true', 'false', 'not given', 'yes', 'no']):
        tips.append(
            f"💡 {q_nums} ask True/False/Not Given — as you read, "
            "actively check whether each claim is confirmed, contradicted, "
            "or simply not mentioned in this paragraph."
        )
    elif any(w in q_texts_combined for w in ['heading', 'title', 'main idea', 'best describes', 'which paragraph']):
        tips.append(
            f"💡 {q_nums} test the main idea — focus on the first and last "
            "sentence of this paragraph; they usually hold the central claim."
        )
    elif any(w in q_texts_combined for w in ['according', 'what', 'when', 'where', 'how many', 'how much', 'who', 'which']):
        # Extract proper-noun keywords from questions
        key_terms: List[str] = []
        for q in related_questions:
            words = re.findall(r'\b[A-Z][a-z]{3,}\b', q.get('text', ''))
            key_terms.extend(words[:2])
        key_terms = list(dict.fromkeys(key_terms))[:4]
        focus = f" Watch for: {', '.join(key_terms)}." if key_terms else ''
        tips.append(
            f"💡 {q_nums} test specific facts.{focus} "
            "Scan for exact numbers, dates, names, and technical terms."
        )
    elif any(w in q_texts_combined for w in ['infer', 'suggest', 'imply', 'purpose', 'attitude', 'tone']):
        tips.append(
            f"💡 {q_nums} test inference — don't just look for literal wording; "
            "think about what the author is implying or their underlying purpose."
        )
    elif any(w in q_texts_combined for w in ['complete', 'fill', 'blank', 'gap']):
        tips.append(
            f"💡 {q_nums} are gap-fill questions — read the surrounding "
            "sentences to predict the missing word before scanning the passage."
        )
    else:
        tips.append(
            f"💡 Read this section carefully to answer {q_nums}. "
            "Look for keywords from the questions that appear in the text."
        )

    # ── Tip 2: Signal-word tip (always added) ─────────────────────
    signal_words = ['however', 'therefore', 'in contrast', 'although',
                    'despite', 'whereas', 'furthermore', 'consequently',
                    'nevertheless', 'on the other hand']
    tips.append(
        "🔍 Watch for signal words like "
        + ', '.join(f'"{w}"' for w in signal_words[:4])
        + " — they often mark the location of answers."
    )

    # ── Tip 3: ADHD-specific attention tip ───────────────────────
    if 'adhd' in all_ld:
        tips.append(
            "⚡ ADHD tip: glance at the question numbers above before "
            "reading — prime your brain on what to hunt for."
        )

    # ── Tip 4: Anxiety calming tip (first section only) ──────────
    if 'anxiety' in all_ld and section_order == 1:
        tips.append(
            "🌀 All answers are in the passage — you don't need prior knowledge. "
            "Take a slow breath before you start reading."
        )

    return tips


def reading_agent_preflight_tips(
    sections: List[Dict],
    questions: List[Dict],
    ld_profile: dict,
) -> Dict[int, List[str]]:
    """Pre-solve all questions and convert insights into per-section reading tips.

    Args:
        sections:   List of section dicts (id, order, heading, body).
        questions:  List of question dicts (id, order, text, group_label).
        ld_profile: Learner LD profile dict.

    Returns:
        Dict mapping section_id -> list of tip strings.
    """
    # Re-use the existing mapper to assign questions to sections
    q_mapping = map_questions_to_paragraphs(sections, questions)

    tips_by_section: Dict[int, List[str]] = {}
    q_by_id = {q['id']: q for q in questions}

    for section in sections:
        sec_id = section['id']
        sec_order = section.get('order', 1)
        related_q_ids = q_mapping.get(sec_id, [])
        related_qs = [q_by_id[qid] for qid in related_q_ids if qid in q_by_id]
        tips_by_section[sec_id] = _infer_tips_for_section(sec_order, related_qs, ld_profile)

    return tips_by_section
