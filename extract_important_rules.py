import re

with open('rulefit_rules.txt', 'r') as f:
    content = f.read()

# Extract rules with importance > 15
rules = []
rule_blocks = content.split('Rule ')[1:]  # Skip header
for block in rule_blocks:
    lines = block.strip().split('\n')
    if len(lines) >= 5:
        importance_line = [l for l in lines if 'Importance:' in l]
        if importance_line:
            importance = float(importance_line[0].split(':')[1].strip())
            if importance > 5:
                rule_line = [l for l in lines if 'Rule:' in l and 'Type:' not in l]
                coef_line = [l for l in lines if 'Coefficient:' in l]
                if rule_line and coef_line:
                    rule = rule_line[0].split('Rule:')[1].strip()
                    coef = float(coef_line[0].split(':')[1].strip())
                    rules.append((importance, coef, rule))

# Sort by importance
rules.sort(reverse=True)
print("Top 50 Most Important Rules:")
print("=" * 80)
for i, (imp, coef, rule) in enumerate(rules[:50]):
    print(f'{i+1:2d}. Importance: {imp:6.1f}, Coef: {coef:8.2f}, Rule: {rule}') 