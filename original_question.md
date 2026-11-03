Engineering Challenge: Aircraft Track Association

Timebox: 6–8 hours
Stack: Your choice of language, framework, and ML libraries.

Objective

Build a small system that associates disconnected aircraft movement fragments using imperfect position reports and supporting data.

Each fragment, called a tracklet, contains observations from one aircraft. Different tracklets may belong to the same aircraft, but the aircraft’s identity is not provided.

A tracklet is a short, connected piece of an aircraft’s recorded path—a series of position reports we already know that belong to the same aircraft.

Your system should identify plausible continuations, accept supported associations, and leave uncertain cases unresolved. It should also demonstrate how additional evidence can change an earlier decision.

We want to understand how you interpret unfamiliar data, compare competing explanations, evaluate associations, and preserve the reasoning behind your results.

Prioritize a defensible baseline and honest evaluation over model complexity. No aviation knowledge or external data source is required.

Input Data

Core position fields follow this structure. Source-specific attributes vary.

{

  "observation_id": "obs-001",

  "tracklet_id": "tracklet-001",

  "observed_at": "2026-01-10T10:00:00Z",

  "received_at": "2026-01-10T10:03:12Z",

  "latitude": 37.7749,

  "longitude": -122.4194,

  "altitude_ft": null,

  "source": "feed-b",

  "attributes": {

    "spd": 246,

    "c2": 92.4,

    "ref": "A1B2C3",

    "q": "2",

    "node": "ingest-17"

  }

}

All top-level fields except altitude_ft and attributes are required. Altitude may be null. Preserve arbitrary JSON attributes.

The example represents one source. Optional attributes may use abbreviated, loosely named, or inconsistent field names across sources. Similar names do not necessarily mean the same thing, and different names may describe related measurements.

The data package includes source notes, known units, and available context. Some optional fields have incomplete documentation or uncertain meaning.

You are expected to inspect distributions, compare related fields, and document your interpretations. Where a field’s meaning cannot be established, explain your assumption or exclude it. You are not required to resolve every field.

The core task is solvable using the documented time and position fields.

Tracklet boundaries are provided and fixed. Each tracklet belongs to exactly one aircraft. Splitting tracklets or identifying real aircraft is outside scope.

Dataset

Use the supplied static JSONL files. No live stream, message broker, or external data collection is required.

The package includes:

Training, validation, and test partitions.
Position reports and supporting streams.
Association labels stored separately from model inputs.
Two staged input snapshots for the additional-evidence demonstration.
Source notes, schema information, and simplified movement assumptions.
 

 

Partitions are separated by simulated scenarios and underlying aircraft. Use the supplied partitions.

 

The data includes:

Clear continuations across gaps.
Multiple plausible continuations.
Nearby aircraft with similar movement.
Tracklets with no valid continuation in the supplied observations.
Identical duplicates and out-of-order delivery.
Irregular sampling and delayed observations.
Missing, redundant, or conflicting supporting values.
Optional fields that are irrelevant to association.
Labels identify each tracklet’s immediate next fragment from the same aircraft within its scenario, or indicate that no successor is present. They do not imply that the aircraft stopped moving.

Source notes describe how supporting records connect to observations or tracklets. Those join relationships are documented even when optional field meanings are uncertain.

Data Preparation

Your pipeline must:

Validate required fields, nonempty IDs, UTC timestamps, and coordinate ranges.
Distinguish observation time from receipt time.
Sort reports by observation time within each tracklet.
Deduplicate identical reports using observation_id.
Reject conflicting content under the same observation ID.
Handle missing altitude and irregular sampling explicitly.
Preserve original reports and source-specific attributes for inspection.
Keep labels separate from predictive inputs.
Explain which supporting fields you use or exclude and why. Distinguish documented meanings from your own interpretations.

Do not silently treat missing values as zero or assume that similarly named fields share units or semantics.

Do not use tracklet IDs, observation IDs, scenario IDs, labels, or other answer-revealing fields as predictive features. IDs may be used for grouping, joining, and referencing outputs.

Source-specific reported identifiers may be used if you justify their availability and reliability. They are not ground-truth aircraft identities.

 

 

Association Requirements

An association A → B means B is the immediate next observed tracklet from the same aircraft among the tracklets supplied for that analysis.

For this exercise:

B must start strictly after A ends.
Each tracklet may have at most one accepted successor and one accepted predecessor.
Accepted associations must not form cycles.
A tracklet may remain unresolved.
An association may span a reporting gap.
A reporting gap alone is not evidence that two tracklets belong together.
Generate plausible candidate pairs using time, position, and any justified supporting signals.

Do not use labels to select candidates. Report candidate-generation recall: the proportion of labeled successor links retained in your candidate set.

A simple strategy for selecting consistent associations is sufficient. Advanced global optimization is not required.

Modeling Requirements

Implement and compare:

A simple baseline using documented rules or a physical continuity score.
One data-driven approach that learns from the training data.
The learned approach may score candidate pairs using a classifier, learned distance, or another justified method. Deep learning is not required.

Both approaches must produce association decisions, including an option to leave a tracklet unresolved.

 

 

For each tracklet, return:

Its accepted successor, if any.
A status distinguishing an accepted association from an unresolved result.
A ranked list of considered successors and their scores.
Supporting measurements for the leading candidates.
A brief reason for accepting or withholding an association.
Document score meaning and direction. Scores do not need to be probabilities. Do not describe them as calibrated confidence unless you evaluate calibration.

Explain how you handle competing candidates and enforce association constraints.

Batch analysis is sufficient. Use only the evidence supplied in each analysis snapshot. Do not access later-stage observations before they are released in the staged demonstration.

Evaluation

Use training data to fit preprocessing and learned components. Use validation data to select features, thresholds, and other decision settings. Evaluate on the test partition without retuning.

Report for both approaches:

Accepted-link precision and recall.
The number of incorrect accepted links.
The proportion of tracklets with a labeled successor that remain unresolved.
Candidate-generation recall.
Results at your chosen operating threshold.
A small comparison showing how precision and recall change under more conservative or permissive acceptance settings.
Evaluate final accepted links across the complete supplied partition, including links excluded during candidate generation. Do not evaluate only sampled candidate pairs.

Incorrect associations are more costly than unresolved cases. Explain how that affects your decision policy.

Include one reproducible stress test, such as reduced sampling, increased position noise, or removal of a supporting signal. Keep the underlying association labels unchanged.

Inspect at least three errors or difficult cases. Include a case where the approaches disagree, if one occurs.

Briefly explain whether the supporting streams improve your results. A justified decision to exclude ambiguous or unhelpful fields is acceptable.

Synthetic performance is not evidence of real-world accuracy. Explain which dataset assumptions simplify the problem and what you would validate with real observations.

Updating Associations

Run the supplied staged scenario:

Analyze the initially available tracklets and supporting records.
Add the supplied later evidence.
Rerun association using the same fitted models and decision settings.
Each snapshot contains only records received by its stated cutoff. Later evidence may include delayed tracklets whose observation times fall within an earlier gap.

Preserve both sets of results and show:

Which associations were accepted or unresolved initially.
Which associations changed after the additional evidence.
The observations and measurements supporting each change.
Any previously accepted link that was withdrawn or replaced.
A revised direct link may reflect a newly available intermediate fragment rather than a change in the inferred aircraft identity. Explain that distinction where applicable.

Do not force a change solely to make the demonstration appear successful. Explain whether the evidence justified the resulting decisions.

Backend Requirements

Endpoint

Behavior

POST /tracklets/associate

Accept a collection of position reports and supporting records; return baseline and learned results.

GET /health

Confirm that the service and required model artifacts are available.

Choose and document request and response schemas. Return consistent JSON errors and appropriate status codes for invalid input.

Training must run through a separate reproducible command. Do not retrain during an inference request.

Save the fitted model, preprocessing configuration, decision settings, and model version so the service can load them after a restart. A database is optional.

Each analysis result must identify its input snapshot, model version, and decision configuration. A saved JSON result and manifest are sufficient.

Required Web Interface

Build a small interface connected to your backend that lets a reviewer:

Select a supplied scenario and run association.
View tracklets in a latitude/longitude plot.
Inspect accepted links, unresolved tracklets, and competing candidates.
Compare baseline and learned results.
Compare the two stages of the additional-evidence scenario.
Inspect supporting measurements and original reports.
See loading, empty, success, and error states.
A simple plot and tables are sufficient. A map provider and elaborate styling are unnecessary.

Demonstration Scenario

Use supplied examples to demonstrate:

A clear continuation.
Competing plausible continuations.
A tracklet left unresolved.
A decision reassessed after additional evidence.
Identical results when input records are reordered or identical duplicates are added.
Submission

Follow the submission checklist in the assessment brief.

Include:

Working preprocessing, candidate generation, baseline, training, and inference code.
Backend and small web interface.
Saved model artifacts or an exact command to generate them.
Automated tests covering validation, duplicates, conflicting IDs, ordering, missing altitude, association constraints, and model save/load consistency.
Evaluation results and failure analysis.
Saved results from both stages of the additional-evidence scenario.
A short account of how you interpreted and selected source-specific fields, including unresolved assumptions.
Exact setup, training, evaluation, testing, and demonstration commands.
Your AI prompt log, if AI tools were used.
 

Everything must run locally without paid services, external credentials, or specialized hardware.

Authentication, deployment, streaming infrastructure, real aircraft identification, tracklet segmentation, and production-scale optimization are outside scope.

Assessment

We will assess data interpretation, association quality, fair evaluation, reproducibility, engineering correctness, and clarity of explanation.

You are assessed on how you investigate uncertain fields and validate assumptions. Using more fields does not earn credit on its own.

Stop at the timebox and document anything unfinished. Prioritize a working baseline, a learned comparison, meaningful evaluation, and a small functional interface over additional features.

 

Clarifications:

These clarifications define the exercise’s assumptions and acceptable simplifications. They supplement the assessment brief, challenge, and data package.

If you encounter a material contradiction or missing definition, ask for clarification. For minor implementation choices, make a reasonable assumption and document it.

Tracklets and Associations

What is guaranteed about a tracklet?

Each tracklet contains observations from exactly one aircraft. Its membership is fixed. You do not need to split tracklets, merge duplicate tracklets, or question whether observations within a tracklet belong together.

Position measurements and optional attributes may still contain errors.

What does an accepted association mean?

A → B means B is the immediate next available tracklet from the same aircraft within that scenario.

It does not identify the aircraft’s real-world registration, ownership, or transponder code.

May associations cross scenario boundaries?

No. Scenarios are independent. Use scenario membership to scope processing, not as a predictive feature.

Can one aircraft have more than two tracklets?

Yes. Accepted links may form chains such as A → B → C. Each tracklet may have at most one accepted predecessor and one accepted successor.

Can tracklets from the same aircraft overlap in time?

Not in this exercise. Candidate B must start strictly after A ends. Other aircraft’s tracklets may overlap.

What does “unresolved” mean?

The system has not accepted a successor from the available candidates. This may mean:

No plausible candidate exists.
Evidence is insufficient.
Several candidates are similarly plausible.
A candidate conflicts with another accepted association.
It is not a claim that the aircraft stopped moving or has no continuation elsewhere.

Must every tracklet be matched?

No. Leaving a case unresolved is acceptable. Incorrectly joining different aircraft is more costly than withholding an association, but leaving everything unresolved does not demonstrate useful association performance.

Data Interpretation

How much ambiguity should we expect in field names?

Core fields have defined meanings. Optional source-specific fields may have abbreviated names, inconsistent conventions, or incomplete documentation.

Use source notes and observed data to form hypotheses. Distinguish confirmed meanings from inferred meanings. You do not need to decode every optional field.

Should we ask what every ambiguous field means?

No. Investigating optional fields is part of the exercise. Ask when an ambiguity prevents you from interpreting a required field or completing the core task.

If an optional field remains unclear, document the uncertainty and decide whether it is appropriate to use.

Can the same attribute name mean different things across sources?

Yes. Interpret optional attributes within their source context. Do not assume a universal schema based on matching names.

Are optional fields guaranteed to help?

No. Some are irrelevant, redundant, unreliable, or only useful under certain conditions. Using more fields does not earn additional credit.

Can we use reported identifiers?

Yes, where supplied, provided you explain their reliability and how they are available at inference time.

Reported identifiers may be missing, change, or be shared incorrectly. They are not authoritative aircraft identities and should not be used as evaluation labels.

Can we use IDs and source names?

Use observation, tracklet, and scenario IDs only for grouping, joining, and referencing results. Do not derive predictive features from their values, formatting, or order.

Source names may be used to select source-specific parsing, units, or reliability handling. Explain any use of source identity as a model feature.

How do supporting records join to position reports?

Use the join keys documented in the data package. Inferring undocumented join relationships is not required.

Timestamps and Record Handling

What is the difference between observed_at and received_at?

observed_at is when the measurement occurred. received_at is when it became available to the platform.

Use observation time for movement continuity. Use receipt time and the supplied snapshot cutoff to determine evidence availability in the staged demonstration.

Do we need to replay records in real time?

No. Static batch processing is sufficient. No Kafka service, replay server, or artificial waiting is required.

How should identical duplicates be compared?

Records with the same observation_id are identical when their parsed JSON content is equal, ignoring object key order and whitespace.

A missing field and an explicitly null field are different. Array order remains significant. Compare original records before feature transformations.

What if the same observation ID has different content?

Reject the conflicting input without silently selecting or overwriting a version. Rejecting the complete analysis request is acceptable if the error identifies the conflicting ID.

Does an unusual measurement make an otherwise valid record invalid?

Not necessarily. Reject schema violations and invalid coordinates or timestamps. Treat physically surprising but structurally valid measurements as data-quality issues to investigate.

Document any filtering or correction, and preserve the original values.

Can we interpolate missing positions or altitude?

Yes, if justified. Preserve the distinction between observed and derived values, and explain the assumptions involved.

You do not need to reconstruct the aircraft’s complete path through a gap.

What movement assumptions may we use?

Use the simplified assumptions supplied with the data. Standard distance, elapsed-time, speed, and direction calculations are sufficient.

You are not expected to research aircraft performance envelopes or implement an aviation-grade motion model.

Modeling and Decisions

Must the learned approach outperform the baseline?

No. A fair comparison and a clear explanation matter more than claiming an improvement.

Report negative results honestly. Explain where the learned method helps, fails, or adds unnecessary complexity.

May both approaches share preprocessing and candidate generation?

Yes. Shared preparation and candidate generation are encouraged where appropriate.

If they use different candidate sets, report candidate-generation recall separately and explain how that affects the comparison.

Does “do not use labels to select candidates” prohibit tuning candidate-generation settings?

No. You may develop candidate-generation rules using training data and select settings using validation data.

At inference time, candidate generation must use only available inputs. Do not insert known true links from label files. Freeze settings before test evaluation.

How sophisticated must association selection be?

A deterministic greedy strategy is acceptable if it respects the constraints and you explain its limitations.

You do not need a graph neural network, advanced assignment solver, or probabilistic tracking framework.

What explanations are required?

Concise measurements and decision reasons are sufficient. For example:

“Candidate B accepted: plausible time gap and projected position; candidate C rejected because it already has a stronger accepted predecessor.”

A natural-language generation system is unnecessary.

Evaluation

What counts as a correct accepted link?

An accepted A → B link is correct only when B matches A’s labeled immediate successor for that evaluated snapshot.

A link that skips an available intermediate tracklet is incorrect, even if both endpoints belong to the same aircraft.

How are precision and recall calculated?

Accepted-link precision: correct accepted links divided by all accepted links.
Accepted-link recall: correct accepted links divided by all labeled successor links.
Candidate-generation recall: labeled successor links retained as candidates divided by all labeled successor links.
Unresolved proportion: tracklets with a labeled successor but no accepted successor divided by all tracklets with a labeled successor.
An incorrect accepted link counts against precision and fails to recover the correct link for recall.

How do we evaluate tracklets labeled as having no successor?

Accepting any successor for those tracklets is an incorrect link. Leaving them unresolved is appropriate.

Do not include them in the denominator for successor-link recall or the unresolved proportion defined above.

What if a metric has a zero denominator?

Report it as undefined or not applicable and include the underlying counts. Do not silently report perfect performance.

May we inspect the supplied test labels?

Use them to calculate final metrics and perform the requested failure analysis after freezing your approach.

Do not use them to select features, tune thresholds, or revise the model. If you make changes after inspecting test results, disclose that those results no longer represent an untouched test evaluation.

How is the internal held-out evaluation different?

The supplied test partition supports your own final evaluation. Govia also retains separate examples and labels for reviewer evaluation.

Neither the baseline nor the learned approach should require labels during inference.

How should we compare thresholds?

Select your operating threshold using validation data. Show the precision–recall tradeoff on validation data.

Report test performance at the frozen operating setting. An additional test threshold comparison is optional and must not be used to choose a new setting.

What is an acceptable stress test?

Apply one reproducible change that preserves tracklet identities and associations, such as removing a supporting field or thinning observations.

Keep every tracklet present and retain enough observations to define it. Explain the change and evaluate without retraining or retuning for that stress test.

Additional Evidence

Is the second snapshot a complete input or an incremental update?

The supplied snapshots are complete inputs. Analyze snapshot 1 independently, then analyze snapshot 2 independently.

You do not need to implement incremental state updates.

Can the first-stage analysis use second-stage evidence indirectly?

No. Features, candidate generation, and decisions for snapshot 1 must use only its available records. Do not compute first-stage features from the combined snapshots.

Keep the fitted models and decision settings unchanged between stages.

Are labels specific to each snapshot?

Yes. The immediate available successor may change when a delayed intermediate tracklet arrives. Use the labels corresponding to the snapshot being evaluated.

Must the system change its answer?

No. Preserve and explain the actual results. A well-supported unresolved decision is acceptable.

If the system does not react to evidence that should distinguish candidates, discuss that limitation.

What must be preserved between runs?

Save both result sets and enough information to identify their input snapshots, model versions, and decision settings.

A full event-sourcing system or database is unnecessary.

Implementation and Delivery

What interface is sufficient?

A small local web interface with a trajectory plot and tables is sufficient. Convenience frameworks are acceptable if the interface calls the required backend.

Manual refresh is sufficient. Authentication, polished styling, and a map provider are unnecessary.

Must results survive a service restart?

The service must reload its saved model and configuration. Persisting previous analysis results in a database is not required; saved result files are sufficient.

What should we do if time runs out?

Stop and document what works, what is unfinished, and your next steps.

Prioritize an end-to-end baseline, a learned comparison, trustworthy evaluation, and a small functional interface. Do not conceal missing functionality.

What AI usage must be disclosed?

If you use AI tools, include the prompts you submitted for this assessment, including follow-up prompts, and briefly explain how you used the outputs.

A text log or tool export is sufficient. AI responses are optional. Do not include credentials or unrelated private information.