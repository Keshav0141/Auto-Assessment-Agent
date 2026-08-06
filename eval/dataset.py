"""Small labeled eval set with genuinely ambiguous/borderline cases.

Each item: (id, question, answer, rubric_text, model_answer, ground_truth_marks).
Rubric text uses the format the parser understands:
  "1 mark for defining X, 2 marks for correct derivation, ..."
"""
ITEMS = [
    # -- Physics: Newton's laws -------------------------------------------------
    dict(
        id="P1", max_marks=5,
        question="State Newton's second law of motion and give its mathematical form. A 2 kg object experiences a net force of 10 N. What is its acceleration?",
        answer="Newton's second law states that the force acting on an object equals the rate of change of momentum. F=ma. For 2kg and 10N, a = F/m = 10/2 = 5 m/s^2.",
        rubric_text="2 marks for stating the law correctly, 1 mark for the correct equation F=ma, 1 mark for correct substitution, 1 mark for correct final value and unit (5 m/s^2)",
        model_answer="Second law: net force equals rate of change of momentum; F=ma. a=F/m=10/2=5 m/s^2. State direction along force.",
        ground_truth=5.0,
    ),
    dict(
        id="P2", max_marks=4,
        question="State Newton's third law of motion and give an everyday example.",
        answer="Every action has an equal and opposite reaction. For example, when you push a wall, the wall pushes you back.",
        rubric_text="2 marks for a correct statement, 2 marks for a correct and clearly explained example",
        model_answer="For every action there is an equal and opposite reaction; forces come in pairs acting on different bodies. Example: recoil of a gun, or a book on a table pushing down and the table pushing up.",
        ground_truth=4.0,
    ),
    dict(
        id="P3", max_marks=6,
        question="A car of mass 1000 kg increases speed from 10 m/s to 20 m/s over 5 s. Calculate (a) the acceleration and (b) the net force acting on it.",
        answer="acceleration = (20-10)/5 = 2 m/s^2. Then force = mass x acceleration = 1000 x 2 = 2000 N.",
        rubric_text="2 marks for using the change in velocity, 2 marks for correct acceleration (2 m/s^2) with unit, 2 marks for correct force (2000 N) with unit",
        model_answer="a=(v-u)/t=(20-10)/5=2 m/s^2. F=ma=1000*2=2000 N.",
        ground_truth=6.0,
    ),
    # Borderline: correct final answer, thin working
    dict(
        id="P4", max_marks=6,
        question="A ball is thrown vertically upward with an initial velocity of 30 m/s. Find (a) the maximum height reached and (b) the time taken to reach it. (g = 10 m/s^2)",
        answer="v^2 = u^2 - 2gh, so h = u^2/2g = 900/20 = 45 m. Time t = u/g = 30/10 = 3 s.",
        rubric_text="2 marks for correct equation choice, 2 marks for correct max height (45 m), 2 marks for correct time (3 s)",
        model_answer="At max height v=0: 0=u^2-2gh -> h=u^2/(2g)=45 m. v=u-gt -> t=u/g=3 s.",
        ground_truth=6.0,
    ),
    # Ambiguous: only part (a) attempted
    dict(
        id="P5", max_marks=6,
        question="A body of mass 5 kg is acted on by a force of 20 N. Find its acceleration. If the body starts from rest, how far does it travel in 4 s?",
        answer="a = F/m = 20/5 = 4 m/s^2.",
        rubric_text="2 marks for correct acceleration, 2 marks for correct distance formula, 2 marks for correct distance (32 m)",
        model_answer="a=F/m=4 m/s^2. s=ut+0.5at^2=0+0.5*4*16=32 m.",
        ground_truth=2.0,
    ),
    # -- Mathematics ------------------------------------------------------------
    dict(
        id="M1", max_marks=4,
        question="Differentiate f(x) = 3x^2 + 2x - 1 with respect to x.",
        answer="The derivative of x^2 is 2x, and derivative of x is 1. So f'(x) = 6x + 2.",
        rubric_text="2 marks for applying the power rule to each term, 1 mark for correct coefficient handling (3x^2 -> 6x), 1 mark for correct final expression (6x + 2)",
        model_answer="f'(x)=6x+2.",
        ground_truth=4.0,
    ),
    dict(
        id="M2", max_marks=5,
        question="Solve the quadratic equation x^2 - 5x + 6 = 0.",
        answer="x^2 - 5x + 6 = 0 factors to (x-2)(x-3) = 0, so x = 2 or x = 3.",
        rubric_text="2 marks for correct factorization, 2 marks for correct solutions, 1 mark for verifying by substitution",
        model_answer="(x-2)(x-3)=0 -> x=2,3.",
        ground_truth=4.0,
    ),
    # Borderline: correct answer, wrong justification path
    dict(
        id="M3", max_marks=4,
        question="Find the integral of 4x^3 with respect to x.",
        answer="Integral of x^3 is x^4/4, times 4 gives x^4, plus constant c. So the answer is x^4 + c.",
        rubric_text="2 marks for correct power rule for integration, 1 mark for correct constant multiplication, 1 mark for adding the constant of integration",
        model_answer="∫4x^3 dx = x^4 + C.",
        ground_truth=4.0,
    ),
    # -- Computer science -------------------------------------------------------
    dict(
        id="C1", max_marks=5,
        question="What is the time complexity of binary search on a sorted array of size n? Explain why.",
        answer="O(log n). Each comparison halves the search space, so after k steps there are n/2^k elements left. Setting this to 1 gives k = log2(n).",
        rubric_text="2 marks for the correct complexity O(log n), 3 marks for a correct explanation of the halving argument",
        model_answer="O(log n). The search interval halves each step; at most log2(n) comparisons.",
        ground_truth=5.0,
    ),
    dict(
        id="C2", max_marks=4,
        question="Explain the difference between an array and a linked list.",
        answer="Arrays store elements in contiguous memory and allow O(1) random access but fixed size. Linked lists use nodes with pointers, allow O(1) insertion/deletion but O(n) access.",
        rubric_text="2 marks for memory layout / access differences, 2 marks for insertion/deletion trade-offs",
        model_answer="Array: contiguous memory, O(1) index access, O(n) insert/delete. Linked list: non-contiguous nodes, O(n) access, O(1) insert/delete at ends.",
        ground_truth=4.0,
    ),
    # Ambiguous: partially correct, conflates concepts
    dict(
        id="C3", max_marks=5,
        question="Explain what a hash table is and analyze the average and worst-case time complexity of search operations in it.",
        answer="A hash table maps keys to indices using a hash function for fast lookup. Average case search is O(1). Worst case is O(n) when all keys hash to the same bucket.",
        rubric_text="2 marks for a correct description of hashing/lookup mechanism, 2 marks for correct average case O(1), 1 mark for correct worst case O(n) with reason",
        model_answer="Hash table: keys hashed to buckets; average search O(1); worst case O(n) under collisions.",
        ground_truth=5.0,
    ),
    # Borderline/incorrect: misses worst-case reason
    dict(
        id="C4", max_marks=5,
        question="Describe how a binary search tree supports insert, delete, and search, and give the average-case time complexity.",
        answer="In a BST, each node has a left and right child. Insert and search compare from the root and go left or right. The average case is O(log n).",
        rubric_text="2 marks for correct structural description, 2 marks for describing search/insert mechanism, 1 mark for correct average case O(log n)",
        model_answer="BST: ordered tree; search/insert/delete compare and recurse left/right; average O(log n), worst O(n).",
        ground_truth=4.0,
    ),
    # -- English / concept definition ------------------------------------------
    dict(
        id="E1", max_marks=3,
        question="Define photosynthesis in your own words.",
        answer="Photosynthesis is the process by which plants convert sunlight into chemical energy, using water and carbon dioxide to make glucose and release oxygen.",
        rubric_text="1 mark for identifying it is a plant process, 1 mark for converting light to chemical energy, 1 mark for mentioning inputs/outputs (CO2+water -> glucose+oxygen)",
        model_answer="Plants use light energy to convert CO2 and water into glucose and oxygen.",
        ground_truth=3.0,
    ),
    # Weak answer: vague, one concept only
    dict(
        id="E2", max_marks=3,
        question="Define photosynthesis in your own words.",
        answer="It is what plants do to make food.",
        rubric_text="1 mark for identifying it is a plant process, 1 mark for converting light to chemical energy, 1 mark for mentioning inputs/outputs (CO2+water -> glucose+oxygen)",
        model_answer="Plants use light energy to convert CO2 and water into glucose and oxygen.",
        ground_truth=1.0,
    ),
    # Borderline: right ideas, sloppy language
    dict(
        id="E3", max_marks=3,
        question="Briefly explain what a machine learning model is.",
        answer="A machine learning model learns patterns from data and uses them to make predictions on new unseen data.",
        rubric_text="1 mark for learning from data, 1 mark for pattern extraction, 1 mark for making predictions on new data",
        model_answer="A program that learns patterns from training data to predict or decide on new inputs.",
        ground_truth=3.0,
    ),
    # Intentionally wrong answer
    dict(
        id="M4", max_marks=4,
        question="Solve the quadratic equation x^2 - 5x + 6 = 0.",
        answer="x = 6 and x = -1, because 6 and -1 multiply to -6 and add to 5.",
        rubric_text="2 marks for correct factorization, 2 marks for correct solutions",
        model_answer="(x-2)(x-3)=0 -> x=2,3.",
        ground_truth=0.0,
    ),
    # -- Expanded adversarial / borderline cases ---------------------------------
    dict(
        id="P6", max_marks=5,
        question="A 2 kg object experiences a net force of 10 N. What is its acceleration? Show all working.",
        answer="a = F/m = 10/2 = 5. So the acceleration is 5 m/s.",
        rubric_text="2 marks for the correct formula F=ma, 2 marks for correct substitution, 1 mark for the correct unit (m/s^2)",
        model_answer="a = F/m = 10/2 = 5 m/s^2.",
        ground_truth=4.0,
    ),
    dict(
        id="P7", max_marks=6,
        question="A ball is thrown vertically upward with an initial velocity of 30 m/s. Find (a) the maximum height reached and (b) the time taken to reach it. (g = 10 m/s^2)",
        answer="Using v^2 = u^2 - 2gh with v=0, h = u^2/2g = 900/20 = 45 m. The time to reach the top is 6 s, because the ball takes 6 s for the whole journey.",
        rubric_text="2 marks for correct equation choice, 2 marks for correct max height (45 m), 2 marks for correct time to reach maximum (3 s)",
        model_answer="v=0 at top: h=u^2/(2g)=45 m. t=u/g=30/10=3 s.",
        ground_truth=4.0,
    ),
    dict(
        id="M5", max_marks=4,
        question="Solve the quadratic equation x^2 - 5x + 6 = 0 and show your factorization.",
        answer="x^2 - 5x + 6 = (x - 2)(x - 3), so the solutions are x = -2 and x = -3.",
        rubric_text="2 marks for correct factorization, 2 marks for correct solutions",
        model_answer="(x-2)(x-3)=0 -> x=2,3.",
        ground_truth=2.0,
    ),
    dict(
        id="M6", max_marks=4,
        question="Find the integral of 4x^3 with respect to x.",
        answer="Using the power rule, the integral of x^3 is x^4/4, and times 4 gives x^4. So the answer is x^4.",
        rubric_text="2 marks for the correct power rule, 1 mark for correct constant multiplication, 1 mark for adding the constant of integration",
        model_answer="∫4x^3 dx = x^4 + C.",
        ground_truth=3.0,
    ),
    dict(
        id="M7", max_marks=5,
        question="Use the quadratic formula to solve x^2 - 6x + 8 = 0.",
        answer="x = (6 ± sqrt(36 + 32)) / 2 = (6 ± sqrt(68)) / 2, so x ≈ 7.1 or x ≈ -1.1.",
        rubric_text="2 marks for correctly identifying a, b, c and substituting, 2 marks for the correct discriminant (4), 1 mark for correct solutions (4, 2)",
        model_answer="a=1, b=-6, c=8; disc = 36-32 = 4; x = (6 ± 2)/2 = 4, 2.",
        ground_truth=2.0,
    ),
    dict(
        id="P8", max_marks=5,
        question="A car travels 100 km in 2 hours at a constant speed. What is its speed in m/s? Show your conversion.",
        answer="Speed = distance / time = 100 / 2 = 50 m/s.",
        rubric_text="2 marks for correct speed in km/h (50), 2 marks for the km/h to m/s conversion factor, 1 mark for the correct value (13.9 m/s)",
        model_answer="50 km/h = 50 * 1000 / 3600 ≈ 13.9 m/s.",
        ground_truth=2.0,
    ),
    dict(
        id="C5", max_marks=5,
        question="Explain how a hash table works and analyze the time complexity of its search operation.",
        answer="A hash table keeps its keys in sorted order so you can search quickly, like a dictionary. Average search is O(log n).",
        rubric_text="2 marks for a correct description of hashing/lookup, 2 marks for correct average case O(1), 1 mark for correct worst case O(n) under collisions",
        model_answer="Keys hashed into buckets; average search O(1); worst case O(n) under collisions.",
        ground_truth=1.0,
    ),
    dict(
        id="C6", max_marks=5,
        question="Describe how a binary search tree supports insert, delete, and search, with time complexities.",
        answer="Search compares from the root and goes left or right, average O(log n). Insert works by comparing and attaching a new node. Deletion works basically the same as insertion.",
        rubric_text="2 marks for correct BST structure and search mechanism, 2 marks for correct insert/delete description, 1 mark for correct average case O(log n)",
        model_answer="BST: ordered tree; search/insert/delete compare and recurse left/right; average O(log n).",
        ground_truth=3.0,
    ),
    dict(
        id="E4", max_marks=3,
        question="Define photosynthesis in your own words.",
        answer="Photosynthesis is how plants make glucose using water and sunlight, and the glucose stores chemical energy for the plant.",
        rubric_text="1 mark for identifying it is a plant process, 1 mark for converting light to chemical energy, 1 mark for mentioning inputs/outputs (CO2+water -> glucose+oxygen)",
        model_answer="Plants use light energy to convert CO2 and water into glucose and oxygen.",
        ground_truth=2.0,
    ),
    dict(
        id="E5", max_marks=3,
        question="Briefly explain what a machine learning model is.",
        answer="It is basically a function that maps inputs to outputs, and it is used to make predictions.",
        rubric_text="1 mark for learning from data, 1 mark for pattern extraction, 1 mark for making predictions on new data",
        model_answer="A program that learns patterns from training data to predict or decide on new inputs.",
        ground_truth=1.0,
    ),
]
