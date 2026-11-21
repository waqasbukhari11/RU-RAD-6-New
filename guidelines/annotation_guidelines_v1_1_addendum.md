# Annotation Guidelines — v1.1 Addendum

**Read this together with the v1.0 guidelines. It sharpens three boundary rules based on the pilot round.**

Pilot result (2 independent annotators, 100 posts): pooled Krippendorff's α = 0.79. Strong: `ig_outgroup` (0.90), `threat` (0.88), `grievance` (0.79). To fix: `dehumanization` (0.64), `mobilization` (0.58). Not yet measurable: `glorification` (only 2 positives in the sample). The three rules below resolve the disagreements we saw.

---

## R1. Dehumanization — insult vs. dehumanization (sharpened)

**Mark `dehumanization = 1` only when the post denies human/moral status to a GROUP, or frames people as vermin/subhuman in a way that invites their exclusion or harm.**

A single animal slur thrown at *one person* in an argument is an **insult**, not dehumanization → **0**.

| Post (pilot) | Label | Why |
|---|---|---|
| `teray jesay kuttey sirf bhonk saktey hein` (dogs like you can only bark) | **0** | animal slur at one individual = insult |
| `IMRANA CHOOHA` (Imran the rat) | **0** | mocking a named person; individual insult |
| `ye log insan nahi, keeray-makoray hain` (these people aren't human, they're insects) | **1** | group cast as subhuman vermin |
| `hum Shiah qoum jaahil nahi hai` (we Shia are not ignorant) | **0** | in-group self-defence — never dehumanization |

**Cursing a specific historical/religious figure** (e.g. "curse on X the dog") is sectarian hostility; mark `dehumanization` only if it casts a **living group** as subhuman, not for condemning one figure.

---

## R2. Mobilization — calling for action vs. reporting it (sharpened)

**Mark `mobilization = 1` only when the post itself urges the audience to take or join contentious collective action** (imperative or direct urging: "come out," "join," "we must march," "rise up").

**Not mobilization (→ 0):**
- Reporting or describing that a protest/struggle is happening or ongoing.
- Quoting someone else's call, unless the poster clearly endorses it as a call to the reader.
- Abstract statements that a struggle "must continue," with no call to the audience.

| Post (pilot) | Label | Why |
|---|---|---|
| `the protest of … people of india` | **0** | reporting a protest, not calling for one |
| `dalit aaj bhi haqooq ki jung lar rahey hein` (Dalits are still fighting for rights) | **0** | describes ongoing struggle; no call to act |
| `… jaddojehad jari rakhna hoga` (struggle must continue) — quoted leader | **0** | quoted general aspiration, not a call to the reader |
| `chup mat baitho, apne haq ke liye niklo` (don't stay silent, come out for your rights) | **1** | direct imperative call to the audience |

---

## R3. Grievance — confirmed rule (reminder + calibration)

The v1.0 rule stands and tested well. Reinforcing the two exclusions that caused all pilot disagreements:

- **Attacking / blaming / taunting a target is NOT grievance** unless the in-group is framed as the victim. ("PDM are traitors," "you lost 93,000 troops" → **0**.)
- **Neutral reporting of an event or others' protest is NOT grievance.** ("All of India is protesting the law," "16,000 rickshaws are on the roads" → **0**.)

**Open judgment call for adjudication — solidarity grievance.** Does grievance for a group the author identifies with but isn't literally part of count?

| Post (pilot) | Proposed | Needs your ruling |
|---|---|---|
| `bharti fouj ne 16 kashmiriyon ko shaheed kiya` (Indian army martyred 16 Kashmiris) | **1** (solidarity) | Confirm: does grievance on behalf of a solidarity group (fellow Muslims/Kashmiris) count as `grievance`? Recommended: **yes**, if the post adopts a victimhood/solidarity frame; **no** if it is bare news reporting. |

Please rule on this so both annotators apply it identically going forward.

---

## R4. `ig_outgroup` — minor clarification (already strong at 0.90)

- An **implied** out-group counts: "they are oppressing Muslims," "minorities are unsafe in [country]" frames a them-vs-us division → `1`.
- Explicitly labelling a named group traitor/ghaddar/enemy → `1` (don't miss these).

---

## R5. Glorification — sampling note (not a rule change)

Only 2 glorification positives appeared in the pilot, so its agreement can't be estimated yet. For the full run we will **oversample glorification-cluster candidates** so there are enough positives both to measure agreement and to train a detector. No definition change; watch for martyrdom/heroism framing of violence and justification of violent acts.

---

*Apply v1.0 + this addendum for the next round. After adjudication of the pilot disagreements and your ruling on R3, this folds into a consolidated v1.1 guideline.*
