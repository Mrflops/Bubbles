file_path = 'settings.txt'
x = int(input("Enter tag size (In meters): "))
y = 2134 #placeholder

with open(file_path, 'r') as file:
    lines = file.readlines()

for i in range(len(lines)):
    if 'FOCAL_LENGTH' in lines[i]:
        lines[i] = 'FOCAL_LENGTH =', x
    elif 'KNOWN_TAG_SIZE' in lines[i]:
        lines[i] = 'KNOWN_TAG_SIZE =', y

with open(file_path, 'w') as file:
    file.writelines(lines)

print("Settings updated.")
