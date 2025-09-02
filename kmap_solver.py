import pandas as pd
import numpy as np
import schemdraw
from schemdraw.parsing import logicparse

def determine_state(var, value):
    if value=='1':
        return var
    else:
        return var+"\'"

def gen_state(var, value):
    state = ""
    for v, va in zip(var,value):
        state+=determine_state(v,va)+" & "

    return state[:-3]

def gen_boolean_map(vars, row_order, col_order):
    b_map = [["" for _ in range(len(col_order))] for _ in range(len(row_order))]
    for row_val in row_order:
        for col_val in col_order:
            b_map[row_order.index(row_val)][col_order.index(col_val)] = gen_state(vars,col_val+row_val)
    
    return pd.DataFrame(b_map)

def infix_to_postfix(expr):
    precedence = {'~':3, '&':2, '|':1}
    tokens = expr.replace('(', ' ( ').replace(')',' ) ').split()
    output, stack = [], []

    for t in tokens:
        if t.isalnum() or t.endswith("'"):
            if(t.endswith("'")):
                stack.append("~")
            output.append(t[0])
        
        elif t in precedence:
            while stack and stack[-1]!='(' and precedence[stack[-1]]>=precedence[t]:
                output.append(stack.pop())
            stack.append(t)
        
        elif t=='(':
            stack.append(t)
        elif t==')':
            while stack and stack[-1]!='(':
                output.append(stack.pop())
            stack.pop()
    
    while stack:
        output.append(stack.pop())
    return output

def postfix_to_verilog(postfix_stack, vars=['A','B']):
    stack = []
    for token in postfix_stack:
        if token.isalnum():
            stack.append(token)
        else:
            if(token=="~"):
                stack.append(token+stack.pop())
            else:
                b = stack.pop()
                stack.append("(" + stack.pop() + " " + token + " " + b + ")")
    
    inputs = ", ".join(f"input {x}" for x in vars)
    y = stack.pop()
    out = f"""module f({inputs}, output Y);
    assign Y = {y};
endmodule"""
    return out

def map2(gray_order,a,b):
    return gray_order.index(str(a)+str(b))

def gen_2_kmap(TRUTH_TABLE):
    kmap = [[-1 for _ in range(2)] for _ in range(2)]

    for a in range(2):
        for b in range(2):
            kmap[b][a] = TRUTH_TABLE[(TRUTH_TABLE['A']==a) & (TRUTH_TABLE['B']==b)].iloc[0,-1]
    df = pd.DataFrame(kmap)
    df.index = ['0','1']
    df.columns = ['0','1']
    return df

def gen_3_kmap(TRUTH_TABLE):
    kmap = [[0 for _ in range(4)] for _ in range(2)]
    gray_code_order = ['00','01','11','10']
    for a in range(2):
        for b in range(2):
            for c in range(2):
                kmap[c][map2(gray_code_order,a,b)] = TRUTH_TABLE[(TRUTH_TABLE['A']==a) & (TRUTH_TABLE['B']==b) & (TRUTH_TABLE['C']==c)].iloc[0,-1]

    kmap_df = pd.DataFrame(kmap,index=['0','1'])
    kmap_df.columns = gray_code_order
    return kmap_df

def gen_4_kmap(TRUTH_TABLE):
    kmap = [[0 for _ in range(4)] for _ in range(4)]
    gray_code_order = ['00','01','11','10']
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    kmap[map2(gray_code_order,c,d)][map2(gray_code_order,a,b)] = TRUTH_TABLE[(TRUTH_TABLE['A']==a) & (TRUTH_TABLE['B']==b) & (TRUTH_TABLE['C']==c) & (TRUTH_TABLE['D']==d)].iloc[0,-1]

    kmap_df = pd.DataFrame(kmap,index=gray_code_order)
    kmap_df.columns = gray_code_order
    return kmap_df

def gen_tb_2():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 4; i = i + 1) begin
            {A, B} = i;
            #5;
        end
    
        $finish;
    end
endmodule"""

    return out

def gen_tb_3():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B, C;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .C(C),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 8; i = i + 1) begin
            {A, B, C} = i;
            #5;
        end
    
        $finish;
    end
endmodule"""

    return out

def gen_tb_4():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B, C, D;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .C(C),
        .D(D),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 16; i = i + 1) begin
            {A, B, C, D} = i;
            #5;
        end
    
        $finish;
    end
endmodule"""

    return out

def gen_tb_5():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B, C, D, E;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .C(C),
        .D(D),
        .E(E),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 32; i = i + 1) begin
            {A, B, C, D, E} = i;
            #5;
        end
    
        $finish;
    end
endmodule"""

    return out

def make_minterms(TRUTH_TABLE,val='1'):
    M = TRUTH_TABLE[TRUTH_TABLE['Y']==val].drop(['Y'],axis=1)
    m,n = M.shape
    minterms = []
    for i in range(m):
        mt = ""
        ones = 0
        for j in range(n):
            mt += str(M.iloc[i,j])
            ones += 1
        if ones>0:
            minterms.append(mt)
    return minterms

def get_prime_implicants(minterms):
    prime_implicants = []
    merges = [False for _ in range(len(minterms))]
    n_merges = 0
    merged_mt, mt1, mt2 = "", "", ""

    for i in range(len(minterms)):
        for c in range(i+1,len(minterms)):
            mt1 = minterms[i]
            mt2 = minterms[c]
            if dashes_align(mt1, mt2) and one_bit_diff_no_dash(mt1, mt2):
                merged_mt = merge_minterms(mt1, mt2)
                if merged_mt not in prime_implicants:
                    prime_implicants.append(merged_mt)
                n_merges += 1
                merges[i] = True
                merges[c] = True

    for j in range(len(minterms)):
        if merges[j]==False and minterms[j] not in prime_implicants:
            prime_implicants.append(minterms[j])
        
    if n_merges==0:
        return prime_implicants
    else:
        return get_prime_implicants(prime_implicants)
    
def one_bit_diff_no_dash(a: str, b: str) -> bool:
  diff = 0
  for c1, c2 in zip(a, b):
      if c1 != c2:
          if c1 == '-' or c2 == '-':
              return False
          diff += 1
          if diff > 1:
              return False
  return diff == 1

def merge_minterms(a: str, b: str) -> str | None:
  if not one_bit_diff_no_dash(a, b):
      return None
  # exactly one real-bit difference → put '-' there
  out = []
  for c1, c2 in zip(a, b):
      out.append('-' if c1 != c2 else c1)
  return ''.join(out)


def dashes_align(mt1,mt2):
    for i in range(len(mt1)):
        if mt1[i]!='-' and mt2[i]=='-':
            return False
    
    return True

def int_expr(mt):
    i = 0
    for c in mt:
        i *= 10
        if c=='-':
            continue
        i += (c-'0')
    return i

def create_implicant_chart(prime_implicants, minterms):
    chart = {}
    for pi in prime_implicants:
        row = "".join("1" if covers(pi, m) else "0" for m in minterms)
        chart[pi] = row
    return chart


def covers(mask: str, mint: str) -> bool:
    return all(mc == '-' or mc == xc for mc, xc in zip(mask, mint))

def select_implicants(chart, minterms):
    selected = []
    covered = set()

    # Step 1: essential implicants
    for j, mt in enumerate(minterms):
        covering = [pi for pi, row in chart.items() if row[j] == "1"]
        if len(covering) == 1:
            epi = covering[0]
            if epi not in selected:
                selected.append(epi)
            for k, bit in enumerate(chart[epi]):
                if bit == "1":
                    covered.add(minterms[k])

    # Step 2 + 3: greedy cover
    while len(covered) < len(minterms):
        # find best PI covering most uncovered
        best_pi = max(
            chart.keys(),
            key=lambda pi: sum(1 for k, mt in enumerate(minterms)
                               if chart[pi][k] == "1" and mt not in covered)
        )
        selected.append(best_pi)
        for k, mt in enumerate(minterms):
            if chart[best_pi][k] == "1":
                covered.add(mt)

    return selected

def make_sop(implicants, vars):
    terms = []
    for implicant in implicants:
        literals = []
        for i, bit in enumerate(implicant):
            if bit == '-':
                continue
            if bit == '1':
                literals.append(vars[i])
            elif bit == '0':
                literals.append(vars[i] + "'")
        # join literals in this implicant with AND
        terms.append(" & ".join(literals))
    # join implicants with OR
    return " | ".join(terms)

def special_case_check(TRUTH_TABLE):
    ys = TRUTH_TABLE["Y"].unique()

    # All 0s (and maybe X)
    if set(ys) <= {"0", "X"}:
        return True, "0"

    # All 1s (and maybe X)
    if set(ys) <= {"1", "X"}:
        return True, "1"

    return False, None

def sop_to_python(expr):
    import re
    expr = re.sub(r"([A-Z])'", r"(not \1)", expr)

    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")

    return expr

def draw_ckt_diag(expr):
    expr = sop_to_python(expr)
    with schemdraw.Drawing(show=False) as d:
        logicparse(expr, outlabel='Y')
    svg = d.get_imagedata('svg').decode()

    svg = svg.replace(
        "<svg ",
        '<svg style="background-color:white;" '
    )
    return svg


def solve_kmap(TRUTH_TABLE, vars):
    mt = make_minterms(TRUTH_TABLE)
    mx = make_minterms(TRUTH_TABLE,val='X')
    prime_implicants = get_prime_implicants(mt+mx)
    chart = create_implicant_chart(prime_implicants,mt)
    ess = select_implicants(chart,mt)
    return make_sop(ess,vars)

def solve_truth_table_2(TRUTH_TABLE):
    kmap = gen_2_kmap(TRUTH_TABLE)
    tb = gen_tb_2()
    special, val = special_case_check(TRUTH_TABLE)
    if special:
        code = f"""module f(input A, input B, output Y);
    assign Y = {"1'b0" if val=='0' else "1'b1"};
endmodule"""
        return kmap, val, code, tb, None
        
    expr = solve_kmap(TRUTH_TABLE,'AB')
    code = postfix_to_verilog(infix_to_postfix(expr))
    img = draw_ckt_diag(expr)
    return kmap, expr, code, tb, img

def solve_truth_table_3(TRUTH_TABLE):
    kmap = gen_3_kmap(TRUTH_TABLE)
    tb = gen_tb_3()

    special, val = special_case_check(TRUTH_TABLE)
    if special:
        code = f"""module f(input A, input B, input C output Y);
    assign Y = {"1'b0" if val=='0' else "1'b1"};
endmodule"""
        return kmap, val, code, tb, None
    expr = solve_kmap(TRUTH_TABLE,'ABC')
    code = postfix_to_verilog(infix_to_postfix(expr),vars=['A','B','C'])
    img = draw_ckt_diag(expr)
    return kmap,expr,code, tb, img

def solve_truth_table_4(TRUTH_TABLE):
    kmap = gen_4_kmap(TRUTH_TABLE)
    tb = gen_tb_4()
    special, val = special_case_check(TRUTH_TABLE)
    if special:
        code = f"""module f(input A, input B, input C, input D, output Y);
    assign Y = {"1'b0" if val=='0' else "1'b1"};
endmodule"""
        return kmap, val, code, tb, None
    expr = solve_kmap(TRUTH_TABLE,'ABCD')
    code = postfix_to_verilog(infix_to_postfix(expr),vars='ABCD')
    img = draw_ckt_diag(expr)
    return kmap, expr, code, tb, img

def solve_truth_table_5(TRUTH_TABLE):
    t0 = TRUTH_TABLE[TRUTH_TABLE['E']=='0'].reset_index(drop=True)
    t1 = TRUTH_TABLE[TRUTH_TABLE['E']=='1'].reset_index(drop=True)

    special, val = special_case_check(TRUTH_TABLE)
    if special:
        kmap = gen_4_kmap(t1 if val=='1' else t0)
        code = f"""module f(input A, input B, input C, input D, input E, output Y);
    assign Y = {"1'b0" if val=='0' else "1'b1"};
endmodule"""
        return ((kmap,None) if val=='0' else (None,kmap)), val, code, tb, None

    k0,_,_ = solve_truth_table_4(t0)
    k1,_,_ = solve_truth_table_4(t1)

    expr = solve_kmap(TRUTH_TABLE,'ABCDE')
    
    code = postfix_to_verilog(infix_to_postfix(expr), vars=list("ABCDE"))
    tb = gen_tb_5()
    img = draw_ckt_diag(expr)
    return (k0,k1), expr, code, tb, img