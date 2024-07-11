file_path = 'settings.txt'
x = int(input("Enter tag size (In meters): "))
y = 700  # Placeholder value

# Read the file content line by line
with open(file_path, 'r') as file:
    lines = file.readlines()

# Modify the specific lines
for i in range(len(lines)):
    if 'KNOWN_TAG_SIZE' in lines[i]:
        lines[i] = f'KNOWN_TAG_SIZE = {x}\n'  # Replace with your desired value
    elif 'FOCAL_LENGTH' in lines[i]:
        lines[i] = f'FOCAL_LENGTH = {y}\n'  # Replace with your desired value

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)

print("Settings updated.")
