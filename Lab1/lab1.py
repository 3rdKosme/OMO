nums = list(map(int, input("Введите числа через пробел: ").split()))

pairs = 0
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] == nums[j]:
            pairs += 1

print("Пар равных элементов:", pairs)


s = input("Введите строку: ")
words = s.split()

longest = max(words, key=len)
print("Самое длинное слово:", longest)
