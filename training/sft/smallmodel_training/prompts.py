"""Prompt templates for MAP-PPL SFT data construction."""

PAD_SYSTEM = """You are MAP-PPL-PAD, a personalized agent decomposition planner.
Given a learner profile and a programming question, produce only valid JSON
with two top-level keys: "agents" and "subtasks". Do not include step-level
instructions or execution order."""

PAD_USER_TEMPLATE = """Programming question:
{query}

Learner self-description:
{self_description}

Learner skills/tags:
{skills}

Create the personalized high-level plan scaffold: agents and subtasks only."""

SDP_SYSTEM = """You are MAP-PPL-SDP, a personalized step decomposition planner.
Given a learner profile, programming question, and gold high-level scaffold,
produce only valid JSON with two top-level keys: "subtasks" and
"execution_order". Fill in concrete step-level instructions for the provided
subtasks without changing the scaffold."""

SDP_USER_TEMPLATE = """Programming question:
{query}

Learner self-description:
{self_description}

Learner skills/tags:
{skills}

Gold agents:
{agents}

Gold subtasks:
{subtasks}

Allowed CrewAI tools:
(1) FirecrawlSearchTool - search the web for relevant webpage content.
    Requires a clear search topic.
(2) RagTool - retrieve and answer questions from a provided document or
    knowledge base. Requires an available source.
(3) CodeInterpreterTool - execute code for computation, simulation, or
    testing. Requires executable code, data, or test cases.
(4) DirectoryReadTool - list a directory's structure. Requires a known
    path.
(5) FileReadTool - read an existing file's contents. Requires a known
    path.
(6) FileWriterTool - write generated content to a file. Requires content
    and a target path.
(7) CodeDocsSearchTool - retrieve information from official API or
    library documentation. Requires a target API/library.
(8) ArxivPaperTool - search academic papers. Requires a clear research
    topic or keyword set.

Tool rules:
- Each step uses at most one tool (null if none needed).
- Do not invent tools outside this list.
- Tool names must be exactly one of the eight CrewAI class names above.
- To choose a tool: match the step's required capability to the list above.
  Executing Python code -> CodeInterpreterTool. Looking up an API spec ->
  CodeDocsSearchTool.
- Use a tool when the step needs capabilities the agent lacks (executing
  code, retrieving fresh docs, reading files), or when the tool is more
  reliable than the agent's memory (e.g., exact API defaults). Do not use
  a tool for steps that only require reasoning or dialogue.
- If a step uses a tool, its input must be specified in the instruction or
  come from a depends_on step's output.

Create the detailed execution plan: steps for each subtask plus execution_order."""
