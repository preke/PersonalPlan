"""
Teaching-domain worker pool for AOP — 8 prototypes mapped 1:1 to the §5
tool pool (Q2 design A). Each worker is the "natural prototype" of one
tool in prompt_for_inference.txt §5, so the AOP meta-agent's worker
choice maps cleanly to a §9 step.agent + step.tool pair.

Mapping (worker → §5 tool):
  ConceptTutor      → None (no tool; pure dialogue / Socratic teaching)
  CodeValidator     → CodeInterpreterTool
  DocsRetriever     → CodeDocsSearchTool
  WebResearcher     → FirecrawlSearchTool
  FileWriter        → FileWriterTool
  PaperSearcher     → ArxivPaperTool
  RagRetriever      → RagTool
  DirectoryReader   → DirectoryReadTool

(FileReadTool from §5 is intentionally not represented as a worker; the
DocsRetriever/RagRetriever cover read-style retrieval needs in the
teaching pool.)

The upstream AOP repo uses ~10 paraphrases per agent for data
augmentation when training the SimilarityMLP reward; we don't train
(option C: LLM-as-judge), so 5 paraphrases per worker is sufficient.
"""

concept_tutor_descriptions = [
    "Conducts Socratic dialogue with the learner — asks diagnostic questions, judges reasoning, never executes code or calls tools.",
    "This worker uses no tool. It explains concepts, surfaces misconceptions via prediction prompts, and gives verbal feedback only.",
    "Pure dialogue agent: walks the learner through ideas, prompts for restatements, and ties new concepts to stated background.",
    "Tutors through guided discovery — sets up predictions, then unveils the correct concept after the learner attempts an answer.",
    "Mentors via targeted questioning that pinpoints knowledge gaps; no lookup, no execution, no file access.",
]

code_validator_descriptions = [
    "This worker uses CodeInterpreterTool to execute code for computation, simulation, or testing. Requires executable code, data, or test cases.",
    "Runs the learner's Python submission against test cases and reports pass/fail outcomes and captured stdout/stderr.",
    "Validates an implementation by invoking CodeInterpreterTool with the learner's most recent code; emits per-case verdicts.",
    "Executes diagnostic scripts on the learner's code to verify that observed behavior matches the destination behavior.",
    "Runs candidate code in a sandbox via CodeInterpreterTool and surfaces failures with concrete error messages.",
]

docs_retriever_descriptions = [
    "This worker uses CodeDocsSearchTool to retrieve information from official API or library documentation. Requires a target API/library.",
    "Looks up authoritative docs for a specific library symbol (signature, defaults, class hierarchy) and returns the relevant section.",
    "Pulls the precise documentation page the next step needs — function signatures, default parameters, deprecation notes.",
    "Retrieves canonical documentation for an API/library so subsequent explanations or validations are anchored in authoritative sources.",
    "Fetches docs for changelog notes, migration guides, and version-specific behavior from official library documentation.",
]

web_researcher_descriptions = [
    "This worker uses FirecrawlSearchTool to search the web for relevant webpage content. Requires a clear search topic.",
    "Researches current third-party guidance — blog posts, RFCs, advisories — that changes faster than official documentation.",
    "Surfaces recent (last 12 months) ecosystem signals on library adoption, anti-patterns, and migration playbooks.",
    "Pulls fresh web articles to supplement official docs when the question is about community practice, not API spec.",
    "Searches the open web via FirecrawlSearchTool for comparison articles and best-practice recommendations the learner needs.",
]

file_writer_descriptions = [
    "This worker uses FileWriterTool to write generated content to a file. Requires content and a target path.",
    "Persists a code snippet, exercise scaffold, or example file to a target path so the learner can open and edit it.",
    "Saves generated content (script, config, dataset) to disk via FileWriterTool when the next step needs the file on the filesystem.",
    "Writes a reproducible example file (e.g., minimum failing script) the learner can run locally to verify a claim.",
    "Materializes generated text into a file at the specified path; downstream steps may then execute or read that file.",
]

paper_searcher_descriptions = [
    "This worker uses ArxivPaperTool to search academic papers. Requires a clear research topic or keyword set.",
    "Surfaces relevant arXiv papers when the question touches on research-grade background (algorithm origins, theoretical results).",
    "Looks up academic publications on a focused research topic to ground an explanation in the primary literature.",
    "Retrieves arXiv abstracts and metadata for keywords the learner gave, so a downstream tutor can summarize them.",
    "Searches arXiv via ArxivPaperTool for papers the learner should read when the topic is at the research frontier.",
]

rag_retriever_descriptions = [
    "This worker uses RagTool to retrieve and answer questions from a provided document or knowledge base. Requires an available source.",
    "Answers a focused question by retrieving from a designated knowledge base (course notes, internal docs, prior chat).",
    "Queries an indexed source via RagTool when the answer is expected to live inside a specific corpus the learner provided.",
    "Pulls grounded passages from a knowledge base and answers a sub-question using only those passages.",
    "Performs retrieval-augmented QA over a provided document set so the response cites in-corpus evidence.",
]

directory_reader_descriptions = [
    "This worker uses DirectoryReadTool to list a directory's structure. Requires a known path.",
    "Inspects the layout of a project directory so downstream steps know where files live before reading or writing them.",
    "Lists files and subdirectories at a target path; useful when the learner asks 'what's in this repo' or 'where is X'.",
    "Enumerates a directory tree via DirectoryReadTool to enable file-aware reasoning in later steps.",
    "Returns a directory's structure (files, subdirs) so the learner or another worker can locate the right path.",
]
