# JobFit Reimplementation Plan

JobFit takes a PDF resume and recommends suitable roles from a catalog of 250 synthetic jobs. We’ll rebuild it module by module. Each lesson covers the concepts, the script to write, the PowerShell commands to run, and the output to expect. You’ll create and run the application code locally as we work through the lessons.

## How the application fits together

| Workflow | Data flow |
| --- | --- |
| Prepare the catalog | Raw JSONL → validated jobs and skill evidence → searchable job text and, for dense search, vectors in Qdrant |
| Match a resume | PDF → page text → candidate profile and query → retrieval → optional reranking → eligibility and scoring → optional explanation → structured recommendations |

The CLI and FastAPI expose the same matching pipeline. BM25 can run against the prepared catalog without loading an embedding model or Qdrant. Retrieval finds jobs worth assessing. Matching applies requirements and checks candidate evidence to decide what to recommend.

## Current checkpoint

Modules 01–08 are complete. Module 09 connects the PDF-to-recommendations workflow through a CLI and FastAPI. A successful BM25 API response matched the saved CLI response. The remaining Module 09 checks cover no matches and an invalid PDF upload. Module 10 will measure retrieval and recommendation quality on labeled cases.

## 01. Configuration and data contracts

**What we’ll build.** Continue the Pydantic `Settings` class for paths and runtime options. Start the shared data contracts with `JobRecord`, `ResumeEvidence`, and `CandidateProfile`. Add `.env.example` to show supported settings and a small setup command to validate the configuration and sample objects.

**Why it matters.** Settings give modules a consistent way to locate data. Models define required fields, allowed values, and how unknown information is represented.

**Checkpoint.** Override a setting through a `JOBFIT_` variable, validate a real job, reject a reversed experience range, and build a candidate fact with a quote and page number.

## 02. Job dataset preparation

**What we’ll build.** Write `app/dataset/dataset_loader.py` to read JSONL line by line, validate jobs, detect duplicate IDs, and collect skill evidence from job descriptions. Write `jobs.jsonl`, `skill_evidence.jsonl`, and `validation_report.json` under the configured output directory.

**Why it matters.** Profiling and search depend on consistent data. A report identifies bad records before they affect matching.

**Checkpoint.** Prepare the supplied dataset and reconcile the input and output counts. Check how the loader reports an invalid line and a duplicate ID.

## 03. Resume parsing and candidate profiling

**What we’ll build.** Extract page text from a PDF with PyMuPDF. Define the parsed resume, preferences, and prepared resume models. Use rules and the corpus’s skill vocabulary to identify candidate facts, keep quotes and page numbers, and build a search query.

**Why it matters.** The query drives retrieval. The profile provides the data and evidence used to assess the candidate against each job.

**Checkpoint.** Prepare one text-based resume, trace a skill to its source page, and check the error for a PDF with no extractable text.

## 04. LangChain documents, embeddings, and Qdrant

**What we’ll build.** Convert jobs into LangChain `Document` objects with searchable text and metadata. Generate dense vectors with BGE-M3, store them in a Qdrant collection, and add commands to build and inspect the index.

**Why it matters.** Dense search can find jobs described with wording different from the query. The catalog’s vectors can be built once and reused across resume requests.

**Checkpoint.** Trace one job through its document, vector, and Qdrant record. Run a search and inspect the returned job ID and metadata.

## 05. BM25, dense, and hybrid retrieval

**What we’ll build.** Implement BM25 keyword search, dense Qdrant search, and filters for location and work mode. Combine the keyword and dense rankings with Reciprocal Rank Fusion (RRF). Expose all three retrieval modes through a shared function.

**Why it matters.** BM25 picks up literal skill names and terms. Dense search captures related wording. RRF combines their results using each job’s rank in the two lists.

**Checkpoint.** Run all three modes for the same query, trace one job’s RRF score, and confirm BM25 runs without the dense index.

## 06. Cross-encoder reranking

**What we’ll build.** Add an optional cross-encoder to score each query and job together. Process shortlisted pairs in batches and keep the reranking scores for inspection.

**Why it matters.** The cross-encoder can compare the texts in more detail. Applying it to a shortlist limits the amount of inference per request.

**Checkpoint.** Compare the order before and after reranking and verify that both lists contain the same job IDs.

## 07. Eligibility, scoring, and recommendations

**What we’ll build.** Check experience, education, and explicit job requirements. Calculate required and preferred skill coverage separately. Record evidence, unknowns, and rejection reasons. Define the final response models and select recommendations using explicit rules.

**Why it matters.** A high retrieval score can surface a job with a hard requirement the candidate fails. Explicit eligibility checks make the recommendation traceable to the available evidence.

**Checkpoint.** Trace an eligible job, a job with a failed requirement, and a job with insufficient evidence. Verify the response when no job qualifies.

## 08. Qwen explanations and semantic evidence review

**What we’ll build.** Add a LangChain prompt and structured output model for Qwen. Generate a readable explanation from the strengths and gaps already recorded by the matcher. Validate the output against the evidence and use a deterministic explanation if generation is disabled or fails. Ask Qwen to suggest resume lines for required skills missed by exact matching.

**Why it matters.** Candidates get readable feedback grounded in the same evidence the matcher used. Suggested resume lines can reveal skills described in different words. The matcher applies its own coverage and eligibility rules.

**Checkpoint.** Compare explanations for the same match with generation on and off. Check that invalid evidence IDs trigger the fallback. Test semantic review with both a clear paraphrase and a partial or unrelated claim, and keep any classification failures visible.

## 09. CLI and FastAPI

**What we’ll build.** Connect resume preparation, retrieval, and matching in a full CLI command. Expose the same flow through FastAPI with a PDF upload endpoint, health and readiness checks, and structured errors.

**Why it matters.** The CLI lets us trace the full workflow locally. The API lets other applications call it using the same business logic.

**Checkpoint.** Run the same resume and settings through both entry points. Compare recommended job IDs and eligibility results, then check an invalid upload.

## 10. Evaluation and testing

**What we’ll build.** Create a labeled suite with relevance and suitability judgments, including no-match cases. Measure retrieval with Precision@K and nDCG@K. Check recommendation quality, no-match outcomes, evidence, and latency.

**Why it matters.** Measuring each stage helps pinpoint where quality breaks down. A suitable job may be missing from the shortlist or rejected later by the matching rules.

**Checkpoint.** Compare all three retrieval modes on the same cases and trace one failure to the stage that produced it.

## 11. Optional comparison without LangChain

**What we’ll build.** Revisit dense retrieval using SentenceTransformers and the Qdrant client directly. Reuse the same sample data and query.

**Why it matters.** Comparing the two versions shows what LangChain handles and what the underlying libraries do.

**Checkpoint.** Trace text preparation, embedding, storage, and search through both implementations and compare the results with equivalent settings.

## Reference

[JobFit Architecture Learning Roadmap](JobFit_Architecture_Learning_Roadmap.md)
