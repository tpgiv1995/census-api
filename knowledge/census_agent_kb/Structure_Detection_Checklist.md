# Structure Detection Checklist
Updated: 2025-10-11T02:57:35.265923Z

Detect using headers + 20-row sample:

- **Dependents layout**
  - Relationship column with EE/EMP, SP, CH → row-based dependents.
  - Repeating Dep1/Dep2 column groups → column-based dependents.

- **Benefit layout**
  - Long: 'Benefit Type' column; repeated employee IDs across rows.
  - Wide: Columns like Medical Plan/Dental Plan/Vision Plan on one row.

- **Identifiers**
  - SSN present? Else use Employee ID/Payroll/Associate ID.
  - Dependent SSN present? If not, note 'no dependent ssn'.

- **Row counts**
  - Rows >> unique employees (and no Dep columns) → long format.

- **Edge cases**
  - Mixed signals (Relationship + Dep columns) → ask user which model to use.
  - Multi-plan slots (Plan 1/2/3) → map first occurrence per benefit.
