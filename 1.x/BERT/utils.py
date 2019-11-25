def str_to_value(input_str):
	# isalpha(): true if the input_str is letter
	# 			 including double-byte characters
	if input_str.isalpha():
		return input_str
	elif input_str.isdigit():
		return int(input_str)
	else:
		return float(input_str)