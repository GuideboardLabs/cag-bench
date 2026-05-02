# Run Summary

- Generated: 2026-05-02 07:40:36
- Source summary: `results/qwen7b_baseline_smoke/20260501_074401/summary_plot_ready.csv`
- Task range: 1 to 30
- Modes detected: rag, dag, cag
- Trials detected: 3
- Total rows analyzed: 270

> *Continuity terms in this dataset are derived from prior tasks' promoted decisions. CAG memory contains those decisions by construction. Disclose this design when comparing CAG to RAG/DAG.*

Primary comparison uses overall means across all rows (all tasks and all trials), not a single last run.

## Composite score
- Overall mean across all tasks/trials:
  - RAG: 27.77 (vs DAG -0.83%; vs CAG -40.20%).
  - DAG: 28.01 (vs RAG +0.84%; vs CAG -39.70%).
  - CAG: 46.45 (vs RAG +67.23%; vs DAG +65.85%).
- Overall ranking by mean: CAG 46.45, DAG 28.01, RAG 27.77.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 0.00, ended at 12.98, change +12.98 points (n/a (start value is 0)), overall increase.
  - DAG: started at 0.00, ended at 26.48, change +26.48 points (n/a (start value is 0)), overall increase.
  - CAG: started at 0.00, ended at 35.68, change +35.68 points (n/a (start value is 0)), overall increase.
- Final-task ranking (mean): CAG 35.68, DAG 26.48, RAG 12.98.

## Checklist quality
- Overall mean across all tasks/trials:
  - RAG: 33.11 (vs DAG -1.62%; vs CAG -21.91%).
  - DAG: 33.66 (vs RAG +1.65%; vs CAG -20.62%).
  - CAG: 42.40 (vs RAG +28.06%; vs DAG +25.98%).
- Overall ranking by mean: CAG 42.40, DAG 33.66, RAG 33.11.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 95.24, ended at 4.76, change -90.48 points (-95.00%), overall decline.
  - DAG: started at 80.95, ended at 38.10, change -42.86 points (-52.94%), overall decline.
  - CAG: started at 95.24, ended at 23.81, change -71.43 points (-75.00%), overall decline.
- Final-task ranking (mean): DAG 38.10, CAG 23.81, RAG 4.76.

## Source evidence recall
- Overall mean across all tasks/trials:
  - RAG: 37.33 (vs DAG -6.71%; vs CAG -2.04%).
  - DAG: 40.02 (vs RAG +7.19%; vs CAG +5.00%).
  - CAG: 38.11 (vs RAG +2.08%; vs DAG -4.77%).
- Overall ranking by mean: DAG 40.02, CAG 38.11, RAG 37.33.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 100.00, ended at 0.00, change -100.00 points (-100.00%), overall decline.
  - DAG: started at 86.67, ended at 66.67, change -20.00 points (-23.08%), overall decline.
  - CAG: started at 100.00, ended at 16.67, change -83.33 points (-83.33%), overall decline.
- Final-task ranking (mean): DAG 66.67, CAG 16.67, RAG 0.00.

## Domain rule recall
- Overall mean across all tasks/trials:
  - RAG: 47.24 (vs DAG +13.38%; vs CAG -16.31%).
  - DAG: 41.67 (vs RAG -11.80%; vs CAG -26.18%).
  - CAG: 56.44 (vs RAG +19.48%; vs DAG +35.47%).
- Overall ranking by mean: CAG 56.44, RAG 47.24, DAG 41.67.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 100.00, ended at 50.00, change -50.00 points (-50.00%), overall decline.
  - DAG: started at 33.33, ended at 16.67, change -16.67 points (-50.00%), overall decline.
  - CAG: started at 66.67, ended at 50.00, change -16.67 points (-25.00%), overall decline.
- Final-task ranking (mean): RAG 50.00, CAG 50.00, DAG 16.67.

## Evidence recall
- Overall mean across all tasks/trials:
  - RAG: 42.29 (vs DAG +3.54%; vs CAG -10.56%).
  - DAG: 40.84 (vs RAG -3.42%; vs CAG -13.61%).
  - CAG: 47.28 (vs RAG +11.80%; vs DAG +15.76%).
- Overall ranking by mean: CAG 47.28, RAG 42.29, DAG 40.84.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 100.00, ended at 25.00, change -75.00 points (-75.00%), overall decline.
  - DAG: started at 60.00, ended at 41.67, change -18.33 points (-30.56%), overall decline.
  - CAG: started at 83.33, ended at 33.33, change -50.00 points (-60.00%), overall decline.
- Final-task ranking (mean): DAG 41.67, CAG 33.33, RAG 25.00.

## Continuity recall
- Overall mean across all tasks/trials:
  - RAG: 17.10 (vs DAG +0.65%; vs CAG -68.47%).
  - DAG: 16.99 (vs RAG -0.65%; vs CAG -68.67%).
  - CAG: 54.24 (vs RAG +217.12%; vs DAG +219.19%).
- Overall ranking by mean: CAG 54.24, RAG 17.10, DAG 16.99.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 42.86, ended at 9.20, change -33.66 points (-78.54%), overall decline.
  - DAG: started at 42.86, ended at 4.60, change -38.26 points (-89.27%), overall decline.
  - CAG: started at 76.19, ended at 45.98, change -30.21 points (-39.66%), overall decline.
- Final-task ranking (mean): CAG 45.98, RAG 9.20, DAG 4.60.

## Token efficiency
- Overall mean across all tasks/trials:
  - RAG: 91.68 (vs DAG +0.73%; vs CAG +20.88%).
  - DAG: 91.02 (vs RAG -0.73%; vs CAG +20.00%).
  - CAG: 75.85 (vs RAG -17.28%; vs DAG -16.67%).
- Overall ranking by mean: RAG 91.68, DAG 91.02, CAG 75.85.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 90.71, ended at 90.71, change +0.00 points (+0.00%), overall flat trend.
  - DAG: started at 90.04, ended at 90.04, change +0.00 points (+0.00%), overall flat trend.
  - CAG: started at 90.02, ended at 58.89, change -31.13 points (-34.58%), overall decline.
- Final-task ranking (mean): RAG 90.71, DAG 90.04, CAG 58.89.

## Latency efficiency
- Overall mean across all tasks/trials:
  - RAG: 77.34 (vs DAG +5.52%; vs CAG +2.35%).
  - DAG: 73.29 (vs RAG -5.23%; vs CAG -3.00%).
  - CAG: 75.56 (vs RAG -2.30%; vs DAG +3.09%).
- Overall ranking by mean: RAG 77.34, CAG 75.56, DAG 73.29.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 81.62, ended at 77.68, change -3.94 points (-4.83%), overall decline.
  - DAG: started at 77.75, ended at 78.75, change +1.00 points (+1.29%), overall increase.
  - CAG: started at 82.19, ended at 60.87, change -21.32 points (-25.94%), overall decline.
- Final-task ranking (mean): DAG 78.75, RAG 77.68, CAG 60.87.

## Memory recall
- Overall mean across all tasks/trials:
  - CAG: 100.00 (no comparisons available).
- Overall ranking by mean: CAG 100.00.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 100.00, ended at 100.00, change +0.00 points (+0.00%), overall flat trend.
- Final-task ranking (mean): CAG 100.00.

## Memory precision
- Overall mean across all tasks/trials:
  - CAG: 39.05 (no comparisons available).
- Overall ranking by mean: CAG 39.05.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 87.50, ended at 39.73, change -47.77 points (-54.60%), overall decline.
- Final-task ranking (mean): CAG 39.73.

## Memory usage rate
- Task progression trend (first task mean to last task mean):

## Continuity per memory token
- Overall mean across all tasks/trials:
  - CAG: 0.28 (no comparisons available).
- Overall ranking by mean: CAG 0.28.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 1.82, ended at 0.07, change -1.75 points (-96.10%), overall decline.
- Final-task ranking (mean): CAG 0.07.

## Contradiction penalty (lower is better)
- Overall mean across all tasks/trials:
  - RAG: 1.50 (vs DAG -25.00%; vs CAG +50.00%).
  - DAG: 2.00 (vs RAG +33.33%; vs CAG +100.00%).
  - CAG: 1.00 (vs RAG -33.33%; vs DAG -50.00%).
- Overall ranking by mean: CAG 1.00, RAG 1.50, DAG 2.00.

- Task progression trend (first task mean to last task mean):
  - RAG: started at 0.00, ended at 0.00, change +0.00 points (n/a (start value is 0)), overall flat trend.
  - DAG: started at 0.00, ended at 0.00, change +0.00 points (n/a (start value is 0)), overall flat trend.
  - CAG: started at 0.00, ended at 5.00, change +5.00 points (n/a (start value is 0)), overall increase.
- Final-task ranking (mean): RAG 0.00, DAG 0.00, CAG 5.00.

## Final Boss: Best Rated Task 30 Answer
- Best score: 37.88
- Mode: CAG
- Trial: 1
- Answer:
```text
### Final Implementation Handoff

#### Phased Delivery Roadmap
1. **Persistence Layer**
   - Implement SQLite storage for DogProfile, HandlerProfile, TrainingGoal, and SessionLog.
   - Ensure UTC timestamps, migration_version, soft_delete, and deterministic IDs.

2. **Domain Model**
   - Define core entities: DogProfile, HandlerProfile, TrainingGoal, and SessionLog.
   - Implement behavior rubrics and safety notes following humane training principles.

3. **Assess Lane**
   - Develop BehaviorRubric with severity 1-5, triggers, reinforcers, and thresholds.
   - Integrate Assess functionality using the defined domain model.

4. **Train Lane**
   - Create TrainingPlan objects with Exercise, Criteria, SafetyNote, and a 7-day microcycle.
   - Implement session states (queued, running, paused, completed, abandoned).
   - Ensure positive reinforcement and humane training practices.

5. **Review Lane**
   - Track RewardLedger entries with reward type and latency.
   - Capture reinforcer and recovery patterns for Review using RewardRecoveryProfile.

6. **Make Outputs**
   - Develop TrainingCard, SessionSummary, ProgressSummary, long-form plans, and short-form cards.
   - Ensure SafetyNote and positive reinforcement boundaries are preserved.

7. **Import/Export**
   - Implement portable JSONL with schema version and ImportExportManifest.
   - Preserve migration_version, soft_delete, encrypted export, Household, DogProfile, and progress history.

8. **Safety Boundaries**
   - Add explicit escalation protocol, high-risk interruption flow, and veterinary escalation triggers.
   - Ensure humane non-diagnostic boundaries are preserved.

9. **Testing**
   - Use scenario fixtures and deterministic seeds for state transitions, privacy tests, import/export tests, safety boundary tests, multi-dog household behavior, and progress review.
   - Expand testing with cross-module integration coverage, migration matrix checks, and regression guardrails mapped to prior feature and safety decisions.

10. **UX Routes**
    - Develop mobile-first UX routes: Dogs tab, Plan tab, Session tab, Progress tab, and Library tab.
    - Map these routes to Assess, Train, Review, and Make outputs.

#### Acceptance Gates
- **Persistence Layer**: Unit tests for SQLite storage with deterministic IDs.
- **Domain Model**: Integration tests for core entities following domain rules.
- **Assess Lane**: End-to-end tests for BehaviorRubric functionality.
- **Train Lane**: Regression tests for TrainingPlan creation and session states.
- **Review Lane**: Validation tests for RewardLedger and RewardRecoveryProfile tracking.
- **Make Outputs**: UI tests for TrainingCard, SessionSummary, ProgressSummary outputs.
- **Import/Export**: Integration tests for JSONL export/import with schema versioning.
- **Safety Boundaries**: Security audits for explicit escalation protocols.
- **Testing**: Automated test coverage reports for all modules.
- **UX Routes**: Usability testing for mobile-first UX routes.

#### Post-Release Validation
- **User Feedback Collection**: Implement a feedback mechanism to gather user insights.
- **Performance Monitoring**: Set up monitoring tools to track app performance and usage patterns.
- **Security Audits**: Conduct regular security audits to ensure compliance with privacy and safety boundaries.
- **Regression Testing**: Perform periodic regression testing to identify and fix any new issues.

### Key Files/Modules
- `persistence/sqlite.py`: SQLite storage implementation.
- `domain/model.py`: Core entities (DogProfile, HandlerProfile, TrainingGoal, SessionLog).
- `assess/rubric.py`: BehaviorRubric implementation.
- `train/session.py`: TrainingPlan and session state management.
- `review/ledger.py`: RewardLedger and RewardRecoveryProfile tracking.
- `make/output.py`: TrainingCard, SessionSummary, ProgressSummary generation.
- `import_export/jsonl.py`: JSONL export/import functionality.
- `safety/boundaries.py`: Safety boundary implementation.
- `testing/scenario.py`: Scenario fixtures for testing.
- `ui/routes.py`: Mobile-first UX route definitions.

### Risks and Mitigation
- **Data Privacy**: Ensure all data is stored locally with deterministic IDs. Implement encryption for sensitive data.
- **Performance Degradation**: Monitor app performance during phased delivery to identify bottlenecks.
- **User Adoption**: Provide comprehensive documentation and support resources to facilitate user adoption.
- **Security Vulnerabilities**: Conduct regular securi

[truncated]
```

## Final Boss: Best Fair-Baseline Task 30 Answer
- Best score: 37.88
- Mode: CAG
- Trial: 1
- Answer:
```text
### Final Implementation Handoff

#### Phased Delivery Roadmap
1. **Persistence Layer**
   - Implement SQLite storage for DogProfile, HandlerProfile, TrainingGoal, and SessionLog.
   - Ensure UTC timestamps, migration_version, soft_delete, and deterministic IDs.

2. **Domain Model**
   - Define core entities: DogProfile, HandlerProfile, TrainingGoal, and SessionLog.
   - Implement behavior rubrics and safety notes following humane training principles.

3. **Assess Lane**
   - Develop BehaviorRubric with severity 1-5, triggers, reinforcers, and thresholds.
   - Integrate Assess functionality using the defined domain model.

4. **Train Lane**
   - Create TrainingPlan objects with Exercise, Criteria, SafetyNote, and a 7-day microcycle.
   - Implement session states (queued, running, paused, completed, abandoned).
   - Ensure positive reinforcement and humane training practices.

5. **Review Lane**
   - Track RewardLedger entries with reward type and latency.
   - Capture reinforcer and recovery patterns for Review using RewardRecoveryProfile.

6. **Make Outputs**
   - Develop TrainingCard, SessionSummary, ProgressSummary, long-form plans, and short-form cards.
   - Ensure SafetyNote and positive reinforcement boundaries are preserved.

7. **Import/Export**
   - Implement portable JSONL with schema version and ImportExportManifest.
   - Preserve migration_version, soft_delete, encrypted export, Household, DogProfile, and progress history.

8. **Safety Boundaries**
   - Add explicit escalation protocol, high-risk interruption flow, and veterinary escalation triggers.
   - Ensure humane non-diagnostic boundaries are preserved.

9. **Testing**
   - Use scenario fixtures and deterministic seeds for state transitions, privacy tests, import/export tests, safety boundary tests, multi-dog household behavior, and progress review.
   - Expand testing with cross-module integration coverage, migration matrix checks, and regression guardrails mapped to prior feature and safety decisions.

10. **UX Routes**
    - Develop mobile-first UX routes: Dogs tab, Plan tab, Session tab, Progress tab, and Library tab.
    - Map these routes to Assess, Train, Review, and Make outputs.

#### Acceptance Gates
- **Persistence Layer**: Unit tests for SQLite storage with deterministic IDs.
- **Domain Model**: Integration tests for core entities following domain rules.
- **Assess Lane**: End-to-end tests for BehaviorRubric functionality.
- **Train Lane**: Regression tests for TrainingPlan creation and session states.
- **Review Lane**: Validation tests for RewardLedger and RewardRecoveryProfile tracking.
- **Make Outputs**: UI tests for TrainingCard, SessionSummary, ProgressSummary outputs.
- **Import/Export**: Integration tests for JSONL export/import with schema versioning.
- **Safety Boundaries**: Security audits for explicit escalation protocols.
- **Testing**: Automated test coverage reports for all modules.
- **UX Routes**: Usability testing for mobile-first UX routes.

#### Post-Release Validation
- **User Feedback Collection**: Implement a feedback mechanism to gather user insights.
- **Performance Monitoring**: Set up monitoring tools to track app performance and usage patterns.
- **Security Audits**: Conduct regular security audits to ensure compliance with privacy and safety boundaries.
- **Regression Testing**: Perform periodic regression testing to identify and fix any new issues.

### Key Files/Modules
- `persistence/sqlite.py`: SQLite storage implementation.
- `domain/model.py`: Core entities (DogProfile, HandlerProfile, TrainingGoal, SessionLog).
- `assess/rubric.py`: BehaviorRubric implementation.
- `train/session.py`: TrainingPlan and session state management.
- `review/ledger.py`: RewardLedger and RewardRecoveryProfile tracking.
- `make/output.py`: TrainingCard, SessionSummary, ProgressSummary generation.
- `import_export/jsonl.py`: JSONL export/import functionality.
- `safety/boundaries.py`: Safety boundary implementation.
- `testing/scenario.py`: Scenario fixtures for testing.
- `ui/routes.py`: Mobile-first UX route definitions.

### Risks and Mitigation
- **Data Privacy**: Ensure all data is stored locally with deterministic IDs. Implement encryption for sensitive data.
- **Performance Degradation**: Monitor app performance during phased delivery to identify bottlenecks.
- **User Adoption**: Provide comprehensive documentation and support resources to facilitate user adoption.
- **Security Vulnerabilities**: Conduct regular securi

[truncated]
```

## Final Boss: Task 30 Comparison (Across Runs)
- RAG: score mean 12.98 (n=3), continuity mean 9.20, checklist mean 4.76, source evidence mean 0.00, domain rule mean 50.00, contradiction penalty mean 0.00.
- DAG: score mean 26.48 (n=3), continuity mean 4.60, checklist mean 38.10, source evidence mean 66.67, domain rule mean 16.67, contradiction penalty mean 0.00.
- CAG: score mean 30.68 (n=3), continuity mean 45.98, checklist mean 23.81, source evidence mean 16.67, domain rule mean 50.00, contradiction penalty mean 5.00.

## Phase Summary
- Source: `phase_summary.csv`

### 1-10
- RAG: composite 35.37, continuity 31.77, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 398.00, contradiction penalty 0.50.
- DAG: composite 34.28, continuity 30.97, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 428.00, contradiction penalty 1.00.
- CAG: composite 48.77, continuity 70.73, memory recall 100.00, memory precision 46.55, memory usage rate nan, continuity/token 0.63, memory tokens 207.70, prompt tokens 628.70, contradiction penalty 0.00.

### 11-20
- RAG: composite 34.27, continuity 15.10, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 367.70, contradiction penalty 2.50.
- DAG: composite 34.41, continuity 15.79, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 397.70, contradiction penalty 4.00.
- CAG: composite 57.72, continuity 59.00, memory recall 100.00, memory precision 35.69, memory usage rate nan, continuity/token 0.16, memory tokens 674.20, prompt tokens 1064.10, contradiction penalty 1.00.

### 21-30
- RAG: composite 13.69, continuity 5.90, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 357.00, contradiction penalty 1.50.
- DAG: composite 15.33, continuity 5.61, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 387.00, contradiction penalty 1.00.
- CAG: composite 32.86, continuity 34.64, memory recall 100.00, memory precision 35.65, memory usage rate nan, continuity/token 0.09, memory tokens 1189.00, prompt tokens 1568.10, contradiction penalty 2.00.

### 20-30
- RAG: composite 14.46, continuity 5.95, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 345.09, contradiction penalty 1.36.
- DAG: composite 16.00, continuity 5.30, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 375.09, contradiction penalty 0.91.
- CAG: composite 35.20, continuity 35.40, memory recall 100.00, memory precision 36.89, memory usage rate nan, continuity/token 0.09, memory tokens 1162.91, prompt tokens 1530.09, contradiction penalty 1.82.

### 30 only
- RAG: composite 12.98, continuity 9.20, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 418.00, contradiction penalty 0.00.
- DAG: composite 26.48, continuity 4.60, memory recall nan, memory precision nan, memory usage rate nan, continuity/token nan, memory tokens 0.00, prompt tokens 448.00, contradiction penalty 0.00.
- CAG: composite 35.68, continuity 45.98, memory recall 100.00, memory precision 39.73, memory usage rate nan, continuity/token 0.07, memory tokens 1410.00, prompt tokens 1850.00, contradiction penalty 5.00.
