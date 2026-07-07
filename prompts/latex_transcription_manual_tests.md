# LaTeX Transcription — Manual Test Batch

Use this file to test an advanced model. For each case:
1. Paste the **system prompt** once at the start of a conversation (or per request).
2. Paste one **user message** at a time.
3. Grade against **grading notes** — errors in student work must appear in LaTeX unchanged.

---

## System prompt

```
You convert word problems and student work into LaTeX math notation.

Rules:
- Transcribe the student's math exactly as they stated or implied it.
- Do NOT fix mistakes, simplify, or "help" with correct math.
- If the student chose the wrong operation or wrong numbers, reflect that in LaTeX.
- Use proper LaTeX syntax (\frac, \text{units}, etc.).
- Output ONLY the LaTeX — no preamble, no commentary.
```

---

## Dev set (22 cases)

### apple_arithmetic_error
**Tags:** wrong_operation, arithmetic, basic

**User message:**
```
Word problem: Maya has 3 bags with 4 apples each. How many apples total?
Student work: "I added 3 + 4 = 7 apples."
```

**Grading notes:** Must show 3+4=7, NOT 3\cdot4=12. Student chose addition wrongly.

---

### ticket_multiplication_skip
**Tags:** wrong_operation, money

**User message:**
```
Word problem: A movie ticket costs $12. Juan buys 5 tickets. How much does he pay?
Student work: "12 + 5 = 17 dollars."
```

**Grading notes:** Preserve 12+5=17. Do not rewrite as 12\cdot5=60.

---

### speed_wrong_formula
**Tags:** rate, wrong_operation

**User message:**
```
Word problem: A car travels 60 miles in 2 hours. What is its average speed?
Student work: "Speed = 60 \times 2 = 120 mph."
```

**Grading notes:** Student multiplied instead of dividing. LaTeX must show 60\cdot2=120.

---

### rectangle_perimeter_area_swap
**Tags:** geometry, wrong_formula, units

**User message:**
```
Word problem: A rectangle is 8 m long and 3 m wide. Find the area.
Student work: "Perimeter formula: 8 + 3 = 11 square meters."
```

**Grading notes:** Wrong formula AND wrong units label. Preserve 8+3=11 and sq m if student said it.

---

### percent_of_wrong_base
**Tags:** percent, arithmetic_error

**User message:**
```
Word problem: A $80 jacket is on sale for 25% off. What is the sale price?
Student work: "25% of 80 means 80 \div 25 = 3.20 off, so sale price is 80 - 3.20."
```

**Grading notes:** Student divided by 25 instead of multiplying by 0.25. Keep their numbers.

---

### fraction_addition_wrong_denominator
**Tags:** fractions, arithmetic_error

**User message:**
```
Word problem: Sam ate 1/2 of a pizza and then 1/3 of the same pizza. How much total?
Student work: "1/2 + 1/3 = 2/5 of the pizza."
```

**Grading notes:** Classic wrong fraction add (added numerators and denominators). Must show 2/5.

---

### linear_equation_sign_error
**Tags:** algebra, sign_error

**User message:**
```
Word problem: You have some money. You spend $15 and have $37 left. How much did you start with?
Student work: "x - 15 = 37, so x = 37 - 15 = 22."
```

**Grading notes:** Sign mishandled in solution step. Keep x-15=37 and x=22 (wrong).

---

### two_step_wrong_isolation
**Tags:** algebra, wrong_operation

**User message:**
```
Word problem: Three friends split a bill equally. Each pays $18. What was the total bill?
Student work: "Let t be total. t \div 3 = 18, so t = 18 \div 3 = 6."
```

**Grading notes:** Should multiply by 3; student divided again. Preserve t=6.

---

### unit_conversion_inverted
**Tags:** units, conversion_error

**User message:**
```
Word problem: A rope is 2.4 meters long. How many centimeters is that?
Student work: "1 m = 100 cm, so 2.4 m = 2.4 \div 100 = 0.024 cm."
```

**Grading notes:** Inverted conversion. LaTeX must show divide by 100, not multiply.

---

### ratio_setup_reversed
**Tags:** ratio, setup_error

**User message:**
```
Word problem: For every 2 cups of flour you need 3 cups of sugar. You have 8 cups of flour. How much sugar?
Student work: "2/3 = 8/x, so x = 8 \cdot 2/3 = 16/3 cups sugar."
```

**Grading notes:** Ratio set up backwards but student proceeded consistently — transcribe their proportion.

---

### negative_temperature_wrong
**Tags:** sign_error, integers

**User message:**
```
Word problem: At midnight it was -4°C. By noon it rose 9 degrees. What is the noon temperature?
Student work: "Noon = -4 + 9 = -13°C."
```

**Grading notes:** Arithmetic sign error (-4+9 should be 5). Must show -13.

---

### probability_wrong_denominator
**Tags:** probability, setup_error

**User message:**
```
Word problem: A bag has 3 red and 5 blue marbles. What is P(red)?
Student work: "P(red) = 3/5 because red is 3 and blue is 5."
```

**Grading notes:** Used blue count as denominator. Keep 3/5, do not fix to 3/8.

---

### quadratic_factoring_wrong
**Tags:** algebra, factoring_error

**User message:**
```
Word problem: A rectangle's area is x^2 + 5x + 6. If width is x+2, write an expression for length.
Student work: "Area = length \cdot width, so length = (x^2+5x+6)/(x+2). I factored top as (x+2)(x+4), so length = x+4."
```

**Grading notes:** Factoring wrong (should be x+3). Preserve student's (x+2)(x+4) cancellation.

---

### exponent_misread
**Tags:** exponential, wrong_operation

**User message:**
```
Word problem: A bacteria colony doubles every hour. Starting with 50, how many after 3 hours?
Student work: "After 3 hours: 50 \cdot 3 = 150 bacteria."
```

**Grading notes:** Linear not exponential. Keep 50\cdot3, not 50\cdot2^3.

---

### geometry_angle_sum_wrong
**Tags:** geometry, fact_error

**User message:**
```
Word problem: Two angles in a triangle are 40° and 65°. Find the third angle.
Student work: "All triangles = 360°, so third = 360 - 40 - 65 = 255°."
```

**Grading notes:** Wrong triangle sum. Preserve 360 and 255.

---

### student_copies_wrong_number
**Tags:** transcription, number_error

**User message:**
```
Word problem: A train travels 240 miles in 4 hours. Find the average speed.
Student work: "I used 204 miles by mistake. Speed = 204/4 = 51 mph."
```

**Grading notes:** Student misread 240 as 204. LaTeX uses 204, not 240.

---

### inequality_written_as_equality
**Tags:** inequality, notation

**User message:**
```
Word problem: You need at least $200 for the field trip. You have $150. How much more do you need?
Student work: "150 + x = 200, so x = 50 dollars more."
```

**Grading notes:** Should be \geq; student used =. Preserve equality form.

---

### irrelevant_number_included
**Tags:** noise, faithful_mapping

**User message:**
```
Word problem: A recipe needs 2 cups of milk for 12 cookies. How much milk for 30 cookies?
Student work: "2/12 = x/30, and also the oven is 350 so 350 + x = something... x = 5 cups."
```

**Grading notes:** Student dragged irrelevant 350 into work. Transcribe their stray equation if stated.

---

### variables_swapped_labels (control)
**Tags:** control, correct_work

**User message:**
```
Word problem: A phone plan costs $30 per month plus $0.10 per text. You sent 80 texts. Total cost?
Student work: "Let t = total, m = texts. t = 0.10m + 30. With m=80: t = 0.10\cdot80 + 30 = 38."
```

**Grading notes:** Correct work — control case. Transcribe faithfully without altering.

---

### word_variable_in_equation
**Tags:** notation, literal_variable

**User message:**
```
Word problem: There are 4 boxes with the same number of pencils. Total pencils 36. How many per box?
Student work: "boxes times pencils = 36, so 4 \cdot pencils = 36, pencils = 9."
```

**Grading notes:** Student used word 'pencils' as symbol. Use \text{pencils} or similar, not x.

---

### mixed_number_parsing_error
**Tags:** fractions, mixed_number

**User message:**
```
Word problem: A board is 2\frac{1}{2} feet long. You cut off \frac{3}{4} foot. How much remains?
Student work: "2\frac{1}{2} - \frac{3}{4} = 2 - 3/4 = 1\frac{1}{4} feet left."
```

**Grading notes:** Student treated mixed number incorrectly. Preserve their steps.

---

### order_of_operations_error
**Tags:** order_of_operations

**User message:**
```
Word problem: Simplify the expression for total cost: 3 + 2 \cdot 4 when buying 3 items at base $3 plus $2 per add-on times 4 add-ons.
Student work: "Total = 3 + 2 \cdot 4 = 5 \cdot 4 = 20."
```

**Grading notes:** Student did (3+2)\cdot4. Keep their grouping error.

---

## Held-out set (22 cases)

### discount_sequential_error
**User message:**
```
Word problem: A $50 item has 20% off, then an extra $5 coupon. Final price?
Student work: "20% off means 50 - 20 = 30, then 30 - 5 = 25 dollars."
```
**Grading notes:** Treated 20% as $20. Preserve 50-20=30.

### average_wrong_divisor
**User message:**
```
Word problem: Scores: 72, 85, 91. Find the average.
Student work: "Average = (72+85+91)/2 = 124 because there are two gaps between three numbers."
```
**Grading notes:** Divided by 2 not 3.

### compound_interest_linearized
**User message:**
```
Word problem: $1000 at 5% annual interest for 2 years, compounded yearly. Final amount?
Student work: "Each year add 5% of 1000, so 1000 + 50 + 50 = 1100."
```
**Grading notes:** Simple interest on compound problem.

### slope_from_two_points_swapped
**User message:**
```
Word problem: A line passes through (2, 5) and (6, 13). Find the slope.
Student work: "m = (2-6)/(5-13) = (-4)/(-8) = 1/2."
```
**Grading notes:** Swapped rise/run.

### system_substitution_arithmetic_slip
**User message:**
```
Word problem: x + y = 10 and x - y = 2. Find x.
Student work: "Add equations: 2x = 12, so x = 6." (Then: "y = 10 - 6 = 3.")
```
**Grading notes:** Include full chain; y should be 4 but student got 3.

### volume_formula_wrong
**User message:**
```
Word problem: A cylinder has radius 3 cm and height 10 cm. Find volume.
Student work: "V = \pi r^2 h = \pi \cdot 3 \cdot 10 = 30\pi cm^3."
```
**Grading notes:** Forgot to square radius.

### scientific_notation_shift_error (control)
**User message:**
```
Word problem: The distance is 4.5 \times 10^6 meters. Write in kilometers.
Student work: "1 km = 1000 m, so divide by 10^3: 4.5 \times 10^6 / 10^3 = 4.5 \times 10^3 km."
```
**Grading notes:** Correct — control case.

### trig_wrong_ratio
**User message:**
```
Word problem: From a point 20 m from a building, the angle of elevation to the top is 35°. Find building height.
Student work: "tan(35°) = adjacent/opposite = 20/h, so h = 20/tan(35°)."
```
**Grading notes:** Inverted opposite/adjacent.

### absolute_value_dropped
**User message:**
```
Word problem: A stock changes by -$8 then +$3. Net change?
Student work: "Net = -8 + 3 = -11."
```
**Grading notes:** Sign error; keep -11.

### piecewise_student_ignores_case
**User message:**
```
Word problem: Parking is $2 for the first hour and $1 each additional hour. Cost for 4 hours?
Student work: "4 hours at $1 per hour = $4."
```
**Grading notes:** Ignored first-hour rate.

### combinatorics_wrong_factorial
**User message:**
```
Word problem: How many ways to arrange 4 distinct books on a shelf?
Student work: "4 + 3 + 2 + 1 = 10 arrangements."
```
**Grading notes:** Added instead of 4!.

### function_composition_reversed
**User message:**
```
Word problem: f(x)=2x+1, g(x)=x^2. Find f(g(3)).
Student work: "f(g(3)) = g(f(3)) = g(7) = 49."
```
**Grading notes:** Swapped composition order.

### log_rule_wrong
**User message:**
```
Word problem: Simplify log_10(100) + log_10(1000).
Student work: "log(100)+log(1000) = log(100\cdot1000) = log(100000) = 5."
```
**Grading notes:** Transcribe student's chain as written.

### student_uses_decimal_wrong_place
**User message:**
```
Word problem: Convert 0.375 to a fraction.
Student work: "0.375 = 375/10 = 37.5/1 = 37\frac{1}{2}."
```
**Grading notes:** Bogus conversion — preserve it.

### dimensionally_inconsistent_equation
**User message:**
```
Word problem: A trip is 120 km at 60 km/h. How many hours?
Student work: "120 km + 60 km/h = 180 hours."
```
**Grading notes:** Added incompatible units.

### inequality_direction_flip
**User message:**
```
Word problem: -2x > 6. Solve for x.
Student work: "Divide by -2: x > -3."
```
**Grading notes:** Forgot to flip inequality.

### geometry_pi_approx_wrong
**User message:**
```
Word problem: Circle radius 7 cm. Circumference?
Student work: "C = 2\pi r = 2 \cdot 3 \cdot 7 = 42 cm."
```
**Grading notes:** Used \pi=3 explicitly.

### expected_value_wrong_weight
**User message:**
```
Word problem: Roll a die. Win $6 on a 6, otherwise lose $1. Expected value?
Student work: "E = (6 + (-1))/2 = 2.5 dollars."
```
**Grading notes:** Wrong probability model.

### vector_add_component_miss
**User message:**
```
Word problem: Walk 3 km east then 4 km north. Displacement magnitude?
Student work: "3 + 4 = 7 km."
```
**Grading notes:** Added legs, not Pythagorean.

### partial_derivative_notation_mix
**User message:**
```
Word problem: Cost C(x,y)=3x+2y with x=10, y=5. Student finds "slope in x".
Student work: "dC/dx = 3 + 2 = 5."
```
**Grading notes:** Mixed partial with total change.

### student_only_verbal_no_equation
**User message:**
```
Word problem: A number increased by 7 is 19. Find the number.
Student work: "I think you subtract 7 from 19 so the number is 26."
```
**Grading notes:** Wrong inverse; transcribe as implied equation n+7=19, n=26.

### multi_step_only_final_wrong
**User message:**
```
Word problem: Length is 5 more than twice a number. Length is 23. Find the number.
Student work: "2x + 5 = 23, so 2x = 28, x = 14."
```
**Grading notes:** 23-5=18 not 28 — preserve student's slip.

---

## Quick rubric (manual grading)

| Score | error_preservation |
|-------|-------------------|
| 0 | Model corrected any student mistake |
| 1 | Fixed one error silently |
| 2 | All student errors preserved exactly |

| Score | latex_validity |
|-------|----------------|
| 0 | Broken LaTeX |
| 1 | Mostly valid |
| 2 | Clean, compilable |

| Score | faithful_mapping |
|-------|-------------------|
| 0 | Wrong problem structure |
| 1 | Partial |
| 2 | Matches student's interpretation |
