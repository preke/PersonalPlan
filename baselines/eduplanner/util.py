"""
Replacement for the missing util.py in the upstream Edu_Planner repo.

Provides two helpers our ported plan.py needs:
- get_students_ability: render a learner persona string from the Skill-Tree.
- map_top_tags_to_levels: heuristic from our main-dataset learner schema
  (about_me + top_tags) to a 5-element ability-level vector aligned with
  ability_tree.json's 5 dimensions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def get_students_ability(ability_tree_path: str, levels: List[int]) -> str:
    """Render a multi-line persona string from per-dimension levels (1-5)."""
    tree = json.loads(Path(ability_tree_path).read_text(encoding="utf-8"))
    abilities = tree["ability_tree"]["ability"]
    lines = []
    for ab, lv in zip(abilities, levels):
        lv = max(1, min(5, int(lv)))
        desc = next(s["Description"] for s in ab["Score"] if s["score"] == lv)
        lines.append(f"- {ab['Name']} (level {lv}/5): {desc}")
    return "Student ability profile:\n" + "\n".join(lines)


def map_top_tags_to_levels(top_tags, about_me: str) -> List[int]:
    """Heuristic: project (top_tags, about_me) onto 5 ability levels.

    The 5 dimensions (aligned with ability_tree.json order) are:
        [0] py    -> Python Idioms          (also general programming languages)
        [1] sql   -> SQL / Data Wrangling   (also generic database/ORM/formats)
        [2] debug -> Debugging & Diagnostics
        [3] algo  -> Algorithms & Data Structures (also system design)
        [4] web   -> Web Frameworks / API Design  (also mobile/devops/HTTP-ish)

    Each tag/keyword bumps the relevant dimension(s) from level 1 (basic) to
    level 2 (intermediate) or level 3 (advanced). about_me prose adds further
    signal via regex for seniority phrases ("senior", "lead", "10 years", ...).

    Fallback: if **no** keyword matches a dimension, that dimension defaults to
    level 2 (neutral intermediate) rather than level 1 — we have no evidence
    the learner is a beginner, so the default should not penalise them.
    """
    text = (about_me or "").lower()
    tags = {(t or "").lower() for t in (top_tags or [])}

    # ------------------------------------------------------------------
    # [0] Python Idioms / general programming-language fluency
    # ------------------------------------------------------------------
    py_lvl2 = {
        # mainstream languages: knowing one = at least intermediate coder
        "python", "python-3.x", "python-2.7", "javascript", "typescript",
        "java", "c#", ".net", "c++", "c", "go", "golang", "rust", "php",
        "ruby", "ruby-on-rails", "swift", "kotlin", "scala", "perl", "r",
        "matlab", "objective-c", "dart", "lua", "bash", "shell", "powershell",
        "vba", "groovy", "haskell", "clojure",
        # basic language features
        "string", "list", "dict", "dictionary", "tuple", "set", "loops",
        "function", "class", "oop", "lambda", "closures",
    }
    py_lvl3 = {
        # idiomatic / advanced language features
        "decorators", "generators", "iterator", "metaclass", "asyncio",
        "concurrency", "multithreading", "multiprocessing", "coroutine",
        "type-hinting", "typing", "dataclasses", "pandas", "numpy", "scipy",
        "tensorflow", "pytorch", "scikit-learn",
    }
    py = 1
    if tags & py_lvl2:
        py = max(py, 2)
    if tags & py_lvl3:
        py = max(py, 3)
    if any(s in text for s in ["senior", "lead", "staff", "principal",
                               "architect", "tech lead", "10 years",
                               "fifteen years", "20 years", "years of experience"]):
        py = max(py, 4)
    if any(s in text for s in ["intern", "junior", "student", "beginner",
                               "new to programming", "learning to code"]):
        py = max(py, 1)  # keep low signal but do not force higher

    # ------------------------------------------------------------------
    # [1] SQL / Data Wrangling — also databases, ORMs, formats, parsing
    # ------------------------------------------------------------------
    sql_lvl2 = {
        "sql", "mysql", "sqlite", "tsql", "oracle", "plsql",
        "database", "database-design", "orm", "sqlalchemy", "hibernate",
        "csv", "json", "xml", "yaml", "excel", "pandas",
        "parsing", "data", "data-cleaning", "etl",
    }
    sql_lvl3 = {
        "postgresql", "mongodb", "redis", "cassandra", "elasticsearch",
        "bigquery", "snowflake", "data-warehouse", "spark", "hadoop",
        "window-functions", "stored-procedures", "indexing", "query-optimization",
    }
    sql = 1
    if tags & sql_lvl2:
        sql = max(sql, 2)
    if tags & sql_lvl3:
        sql = max(sql, 3)
    if any(s in text for s in ["data engineer", "data scientist", "dba",
                               "database administrator", "etl", "analytics"]):
        sql = max(sql, 4)

    # ------------------------------------------------------------------
    # [2] Debugging & Diagnostics — errors, logging, profiling, testing
    # ------------------------------------------------------------------
    debug_lvl2 = {
        "debugging", "exception", "error-handling", "try-catch", "stack-trace",
        "logging", "log4j", "log4net", "console", "warnings",
        "testing", "unit-testing", "unit-test", "junit", "pytest", "jest",
        "mocha", "mocking", "tdd", "integration-testing",
        "printing", "console-application",
    }
    debug_lvl3 = {
        "performance", "profiling", "memory-leaks", "garbage-collection",
        "valgrind", "gdb", "pdb", "lldb", "strace", "perf",
        "benchmarking", "monitoring", "observability", "tracing",
    }
    debug = 1
    if tags & debug_lvl2:
        debug = max(debug, 2)
    if tags & debug_lvl3:
        debug = max(debug, 3)
    if any(s in text for s in ["sre", "site reliability", "devops",
                               "linux", "production", "on-call", "incident"]):
        debug = max(debug, 4)

    # ------------------------------------------------------------------
    # [3] Algorithms & Data Structures — also system design / architecture
    # ------------------------------------------------------------------
    algo_lvl2 = {
        "arrays", "array", "list", "linked-list", "hashmap", "hashtable",
        "stack", "queue", "sorting", "search", "binary-search",
        "recursion", "iteration", "string-matching",
    }
    algo_lvl3 = {
        "algorithms", "algorithm", "data-structures", "dynamic-programming",
        "dp", "graph", "graph-algorithm", "tree", "binary-tree", "trie",
        "heap", "priority-queue", "complexity", "big-o", "leetcode",
        "competitive-programming", "greedy", "backtracking", "divide-and-conquer",
        "design-patterns", "architecture", "microservices", "scalability",
        "distributed", "distributed-systems", "system-design",
        "machine-learning", "deep-learning", "neural-network", "nlp",
    }
    algo = 1
    if tags & algo_lvl2:
        algo = max(algo, 2)
    if tags & algo_lvl3:
        algo = max(algo, 3)
    if any(s in text for s in ["computer science", "phd", "research",
                               "ms in cs", "msc", "algorithmic", "ml engineer"]):
        algo = max(algo, 4)

    # ------------------------------------------------------------------
    # [4] Web Frameworks / API Design — also frontend, mobile, devops, HTTP
    # ------------------------------------------------------------------
    web_lvl2 = {
        # frontend basics
        "html", "html5", "css", "css3", "jquery", "dom", "bootstrap",
        "ajax", "forms", "form", "page-break", "pdf",
        # mobile basics
        "ios", "android", "iphone", "ipad", "uikit",
        # HTTP / data
        "http", "https", "url", "cookies", "session", "cors",
        "image", "image-processing", "file", "file-io", "upload", "download",
        # backend basics
        "apache", "iis", "nginx", "tomcat", "express", "node.js", "nodejs",
        "spring", "spring-boot", "asp.net", "asp.net-mvc",
    }
    web_lvl3 = {
        # advanced frameworks / SPA
        "django", "fastapi", "flask", "rails", "laravel", "symfony",
        "react", "react.js", "reactjs", "vue", "vue.js", "vuejs", "angular",
        "angularjs", "svelte", "next.js", "nuxt.js",
        "react-native", "flutter", "ionic", "xamarin",
        # APIs / protocols
        "rest", "rest-api", "restful", "graphql", "grpc", "soap",
        "webservices", "web-api", "api", "openapi", "swagger",
        "oauth", "jwt", "authentication", "authorization",
        # devops / cloud
        "docker", "kubernetes", "k8s", "ci", "ci-cd", "jenkins", "github-actions",
        "deployment", "aws", "amazon-web-services", "gcp", "google-cloud",
        "azure", "heroku", "terraform", "ansible",
        "regex", "regular-expression",
    }
    web = 1
    if tags & web_lvl2:
        web = max(web, 2)
    if tags & web_lvl3:
        web = max(web, 3)
    if any(s in text for s in ["backend", "frontend", "full stack", "fullstack",
                               "full-stack", "web developer", "mobile developer",
                               "devops", "cloud engineer", "platform engineer"]):
        web = max(web, 4)

    # ------------------------------------------------------------------
    # Neutral fallback: if a dimension is still at level 1 (no signal),
    # bump it to level 2 (intermediate). We have no evidence the learner
    # is a beginner, so a neutral default is fairer than the lowest level.
    # ------------------------------------------------------------------
    levels = [py, sql, debug, algo, web]
    levels = [max(2, lv) for lv in levels]

    # But if about_me explicitly self-identifies as beginner/student, allow
    # the lowest level back through (override the neutral fallback).
    if any(s in text for s in ["beginner", "just starting", "first time",
                               "new to programming", "learning to code",
                               "complete novice"]):
        levels = [py, sql, debug, algo, web]  # raw, un-floored

    return levels
