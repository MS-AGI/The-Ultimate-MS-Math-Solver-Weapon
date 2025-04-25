import streamlit as st
from sympy import *
from sympy.abc import x, y, z, n
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# Initialize SymPy latex 
init_printing(use_latex=True)

# Set up Streamlit page configuration
st.set_page_config(layout="wide")
st.title("🧮 Ultimate ADVANCED Math Calculator")

# Initialize session state variables
if "expr_input" not in st.session_state:
    st.session_state.expr_input = "sinh(x)**2 + cosh(x)**2"
if "history" not in st.session_state:
    st.session_state.history = []
if "operation" not in st.session_state:
    st.session_state.operation = "Simplify"
if "reload_trigger" not in st.session_state:
    st.session_state.reload_trigger = False
if "show_mathml" not in st.session_state:
    st.session_state.show_mathml = False

# If a reload is triggered, reload the page
if st.session_state.reload_trigger:
    st.session_state.reload_trigger = False
    st.rerun()

# Input area for expression
expr_input = st.text_area("Enter Expression (LaTeX-like or SymPy syntax):", value=st.session_state.expr_input, height=100, key="expr_input_box")

# Update session state with the current input
st.session_state.expr_input = expr_input

# Select operation dropdown
operation = st.selectbox("Select Operation", [
    "Simplify", "Evaluate Limit", "Derivative", "Nth Derivative", "Integral",
    "Nth Integral", "Closed Integral", "Multiple Integral", "Solve Equation",
    "Trigonometric Identity", "2D Plot", "3D Plot", "Probability", "Trigonometric Evaluation", "GCD / LCM"
])

# Update selected operation in session state
st.session_state.operation = operation

# Variable to store results
result = ""

try:
    # Parse the entered expression
    raw_expr = st.session_state.expr_input.strip()

    # Handling different operations
    expr = sympify(raw_expr)

    if operation == "Simplify":
        try:
            # First attempt regular simplification
            simplified = simplify(expr)

            # Expand if it's a power of sum or product of expressions
            if expr.is_Pow or expr.is_Mul or expr.is_Add:
                expanded = expand(expr)
                result = simplified if simplified != expr else expanded
            else:
                result = simplified

        except Exception as e:
            result = f"Simplify error: {e}"


    elif operation == "Derivative":
        result = diff(expr, x)

    elif operation == "Nth Derivative":
        nth = st.number_input("Order of Derivative (n)", min_value=1, value=3)
        result = diff(expr, x, nth)

    elif operation == "Integral":
        result = integrate(expr, x)

    elif operation == "Nth Integral":
        nth = st.number_input("Order of Integral (n)", min_value=1, value=2)
        temp = expr
        for _ in range(nth):
            temp = integrate(temp, x)
        result = temp

    elif operation == "Closed Integral":
        # Ask for the interval for closed integral
        lower_limit = st.number_input("Enter lower limit:", value=0)
        upper_limit = st.number_input("Enter upper limit:", value=2 * math.pi)
        result = integrate(expr, (x, lower_limit, upper_limit))

    elif operation == "Multiple Integral":
        # Ask for the interval for multiple integrals
        lower_limit_x = st.number_input("Enter lower limit for x:", value=0)
        upper_limit_x = st.number_input("Enter upper limit for x:", value=2)
        
        lower_limit_y = st.number_input("Enter lower limit for y:", value=0)
        upper_limit_y = st.number_input("Enter upper limit for y:", value=2)
        
        # For multiple integrals, check if more than one variable is involved
        if len(expr.free_symbols) == 1:
            result = integrate(expr, (x, lower_limit_x, upper_limit_x), (y, lower_limit_y, upper_limit_y))  # Double integral
        elif len(expr.free_symbols) == 2:
            result = integrate(expr, (x, lower_limit_x, upper_limit_x), (y, lower_limit_y, upper_limit_y))  # Double integral
        elif len(expr.free_symbols) == 3:
            lower_limit_z = st.number_input("Enter lower limit for z:", value=0)
            upper_limit_z = st.number_input("Enter upper limit for z:", value=2)
            result = integrate(expr, (x, lower_limit_x, upper_limit_x), (y, lower_limit_y, upper_limit_y), (z, lower_limit_z, upper_limit_z))  # Triple integral
        else:
            result = "Multiple integrals not supported for this expression"

    elif operation == "Solve Equation":
        result = solve(expr, x)

    elif operation == "Trigonometric Identity":
        result = trigsimp(expr)

        # Handling Trigonometric Evaluation
    if operation == "Trigonometric Evaluation":
        trig_op = st.selectbox("Select Trigonometric Operation", [
            "Sin^n(θ)", "Cos^n(θ)", "Tan^n(θ)", "Sinh^n(x)", "Cosh^n(x)",
            "Inverse Sin (asin)", "Inverse Cos (acos)", "Inverse Tan (atan)"
        ])

        if trig_op == "Sin^n(θ)":
            power = st.number_input("Enter the power n", min_value=1, value=2)
            angle = st.text_input("Enter the angle in terms of π (e.g., π/3)", value="π/200")
            angle_expr = sympify(angle)
            if angle == "pi":
                result = sin(angle_expr)**power
            else:
                result = math.sin(angle_expr)**power

        elif trig_op == "Cos^n(θ)":
            power = st.number_input("Enter the power n", min_value=1, value=2)
            angle = st.text_input("Enter the angle in terms of π (e.g., π/3)", value="π/200")
            angle_expr = sympify(angle)
            if angle == "pi":
                result = cos(angle_expr)**power
            else:
                result = math.cos(angle_expr)**power

        elif trig_op == "Tan^n(θ)":
            power = st.number_input("Enter the power n", min_value=1, value=2)
            angle = st.text_input("Enter the angle in terms of π (e.g., π/3)", value="π/200")
            angle_expr = sympify(angle)
            if angle == "pi":
                result = tan(angle_expr)**power
            else:
                result = math.tan(angle_expr)**power

        elif trig_op == "Sinh^n(x)":
            power = st.number_input("Enter the power n", min_value=1, value=2)
            angle = st.text_input("Enter the angle in terms of π (e.g., π/3)", value="π/200")
            angle_expr = sympify(angle)
            if angle == "pi":
                result = sinh(angle_expr)**power
            else:
                result = math.sinh(angle_expr)**power

        elif trig_op == "Cosh^n(x)":
            power = st.number_input("Enter the power n", min_value=1, value=2)
            angle = st.text_input("Enter the angle in terms of π (e.g., π/3)", value="π/200")
            angle_expr = sympify(angle)
            if angle == "pi":
                result = cosh(angle_expr)**power
            else:
                result = math.cosh(angle_expr)**power

        elif trig_op == "Inverse Sin (asin)":
            value = st.number_input("Enter value (e.g., 0.5)", min_value=-1.0, max_value=1.0, value=0.5)
            result = asin(value)

        elif trig_op == "Inverse Cos (acos)":
            value = st.number_input("Enter value (e.g., 0.5)", min_value=-1.0, max_value=1.0, value=0.5)
            result = acos(value)

        elif trig_op == "Inverse Tan (atan)":
            value = st.number_input("Enter value (e.g., 1)", value=1)
            result = atan(value)

    elif operation == "2D Plot":
        p = plot(expr, show=False)
        buf = BytesIO()
        p.save(buf)
        st.image(buf)

    elif operation == "3D Plot":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_vals = y_vals = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = lambdify((x, y), expr, 'numpy')(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis')
        st.pyplot(fig)
    
    elif operation == "GCD / LCM":
        gcd_lcm_op = st.selectbox("Select Operation", ["GCD (HCF / GCF)", "LCM"])
        nums_input = st.text_input("Enter numbers separated by commas", value="12,18,24")
        try:
            nums = list(map(int, nums_input.split(",")))
            if gcd_lcm_op == "GCD (HCF / GCF)":
                result = math.gcd(nums[0], nums[1])
                for num in nums[2:]:
                    result = math.gcd(result, num)
            elif gcd_lcm_op == "LCM":
                def lcm(a, b):
                    return abs(a*b) // math.gcd(a, b)
                result = nums[0]
                for num in nums[1:]:
                    result = lcm(result, num)
        except:
            result = "Invalid input. Please enter integers separated by commas."


    elif operation == "Probability":
        prob_op = st.selectbox("Select Probability Operation", [
            "Basic Probability", "Factorial", "Combinations (nCr)", "Permutations (nPr)",
            "Union (A∪B)", "Intersection (A∩B)", "Conditional Probability (P(A|B))"
        ])
        if prob_op == "Basic Probability":
            prob_var = st.text_input("Enter Total Outcomes and Favorable Outcomes (format: total,fav)", value="6,1")
            try:
                total, fav = map(int, prob_var.split(","))
                result = Rational(fav, total)
            except:
                result = "Invalid input format. Use 'total,fav'"
        elif prob_op == "Factorial":
            val = st.number_input("Enter n for n!", min_value=0, value=5)
            result = factorial(val)
        elif prob_op == "Combinations (nCr)":
            n_val = st.number_input("n (total elements)", min_value=0, value=5)
            r_val = st.number_input("r (selecting)", min_value=0, value=3)
            result = binomial(n_val, r_val)
        elif prob_op == "Permutations (nPr)":
            n_val = st.number_input("n (total elements)", min_value=0, value=5)
            r_val = st.number_input("r (selecting)", min_value=0, value=3)
            result = factorial(n_val) / factorial(n_val - r_val)
        elif prob_op == "Union (A∪B)":
            pa = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5)
            pb = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.5)
            pab = st.number_input("P(A ∩ B)", min_value=0.0, max_value=1.0, value=0.2)
            result = pa + pb - pab
        elif prob_op == "Intersection (A∩B)":
            pa = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5)
            pb = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.5)
            result = pa * pb
        elif prob_op == "Conditional Probability (P(A|B))":
            pab = st.number_input("P(A ∩ B)", min_value=0.0, max_value=1.0, value=0.2)
            pb = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.5)
            result = pab / pb if pb != 0 else "Undefined (P(B) = 0)"

    # Display result in LaTeX
    if isinstance(result, (Expr, int, float, list, Rational)):
        st.latex(latex(result))
        if operation=="Trigonometric Evaluation":
            d=N(result)
            st.write("Decimal Value")
        if st.checkbox("Show MathML Output"):
            st.code(mathml(result, printer='presentation'), language='xml')

    # Add the current expression and result to history
    history_entry = {
        "expression": raw_expr,
        "operation": operation,
        "result": latex(result)
    }
    st.session_state.history.insert(0, history_entry)

except Exception as e:
    st.error(f"Error: {e}")
