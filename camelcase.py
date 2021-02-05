import re

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def camelCaseSeparation(words, variableName):
	capital_letter = capitalize_words(words)
	lower = lower_words(words)
	all_words = capital_letter.union(lower)

	variable_words = camel_case_split(variableName)

	for word in variable_words:
		if word not in all_words:
			return False
	return True

def capitalize_words(words):
	upper_set = set()
	for word in words:
		upper_set.add(word.capitalize())
	return upper_set

def lower_words(words):
	lower_set = set()
	for word in words:
		lower_set.add(word.lower())
	return lower_set

words = ["is", "valid", "right"]
test = "isValId"
"""
test_split = camel_case_split(test)
lower_split = lowercase(test_split)

print(camel_case_split(test))
print(test_split)
print(lower_split)
"""
print(camelCaseSeparation(words, test))