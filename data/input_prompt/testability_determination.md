### **Task Description**
You are a **software requirements modeling and testing expert**, specialized in determining whether a natural language rule and its formalized requirement are testable.
Now, you need to determine whether a given rule is a **Testable Requirement**.

The input of a rule includes:
1. **Natural Language Rule**
2. **Formalized Rule**

You need to determine whether the rule satisfies both the "**Requirement Validity**" and "**Testability**" criteria, and finally output:
* `True` (testable requirement) or `False` (not testable or not a requirement)
* If `False`, provide the reason.

---

### **Input Format**
The natural language description of the rule, e.g.: The trading system shall complete matching within 5 seconds after receiving a valid order.
The corresponding formalized rule, e.g.: rule 1\nif Actor = TradingSystem and Event = OrderReceived and Time = Within5s and Action = CompleteMatching\nthen Result = Success

---

### **Judgment Criteria**
#### 1. Requirement Validity
The rule must indeed be a requirement, rather than explanatory, definitional, background, or non-constraining text. That is, it should express a constraint on the system or behavior, containing a "when...then..." or "the system shall..." structure.

Criteria:
* Contains an **explicit behavioral constraint** (shall / must / prohibited, etc.)
* Involves elements such as **condition + operation + result**
* Not a term definition, background explanation, or referential clause

#### 2. Testability
Satisfies the following **testability criteria defined by OCL constraints**:

1. **StructuralCompleteness**
   ```
   context Rule
   inv StructuralCompleteness:
       not self.Precondition.oclIsUndefined() and
       not self.Operation.oclIsUndefined() and
       not self.ExpectedResult.oclIsUndefined()
   ```
   -> The three elements condition, action, and result are all present.

2. **RuleElementDeterministic**
   ```
   context Rule
   inv RuleElementDeterministic:
       self.Precondition->forAll(e | e.concrete() and e.deterministic())
       self.Operation->forAll(e | e.concrete() and e.deterministic())
       self.ExpectedResult->forAll(e | e.concrete() and e.deterministic())
   ```
   -> Each element is concrete and uniquely interpretable, with no vagueness or ambiguity.

3. **ActionExecutable**

   ```
   context Rule
   inv ActionExecutable:
       self.Operation.Action.notEmpty() and
       self.Operation.Action.executable()
   ```
   -> The action can be executed by the system or the user.

4. **ExpectedResultObservable**

   ```
   context Rule
   inv ExpectedResultObservable:
       self.ExpectedResult.Result.notEmpty() and
       self.ExpectedResult.Result.observable()
   ```
   -> The result can be detected and verified in the system.

5. **DeterministicOutcome**

   ```
   context Rule
   inv DeterministicOutcome:
       Rule.allInstances()->forAll(r2 |
           if r2 <> self and r2.Precondition = self.Precondition and r2.Operation = self.Operation
           then r2.ExpectedResult.ResultStatus = self.ExpectedResult.ResultStatus
           else true
           endif)
   ```
   -> For any two rules, the same precondition + operation must not yield mutually contradictory expected results (unless the two results can coexist, e.g., they have an "or" relationship).

---

### **Output Format**
Output true/false, indicating whether the rule is a testable requirement.
If the rule is not testable, explain the reason in one sentence.

---

### **Reasoning Strategy (order of thinking)**
1. First determine whether it is a **requirement** (excluding background explanations, definitions, etc.).
2. Then check the 5 testability constraints one by one.
3. If all are satisfied, output `True`; otherwise `False`.
4. The reason should be concise and verifiability-oriented.

### **Some intuitive judgment guidelines**:
1. We consider that any "may..." implies that both doing and not doing are successful, which does not affect the testability judgment;
2. All elements of a testable requirement must be deterministic; expressions such as "other...", "unless otherwise specified...", or "the submission time when responding (rather than a specific 9:00-10:00)" must not appear;
3. A testable requirement must not contain referential content (such as "Article ...", "the previous chapter", "the preceding two rules", etc.);
4. The formalized rule has already undergone some testability processing and has stronger testability than the original rule; when judging, it is mainly sufficient to determine whether the formalized rule satisfies the testability requirements, and the original rule text is only a reference.
Note: All testability judgments use the strict OCL constraint mode, make no contextual assumptions, and favor a "purely textual testability" judgment (must be dependency-free and directly verifiable).

---

### **Example**
#### Input:
When the system receives a user's payment instruction, it shall immediately deduct the corresponding account balance and return the transaction result status.
rule 1
if Actor = System and Event = UserPaymentInstructionReceived and Action = Deduct and OperationPart = AccountBalance and Action = Return and OperationPart = TransactionSuccessStatus
then Result = Success and ResultStatus = TransactionSuccess

#### Output:
true
