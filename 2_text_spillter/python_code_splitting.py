from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return f"{amount} deposited. New balance: {self.balance}"

    def withdraw(self, amount):
        if amount > self.balance:
            return "Insufficient balance!"
        self.balance -= amount
        return f"{amount} withdrawn. Remaining balance: {self.balance}"

    def get_balance(self):
        return f"Account owner: {self.owner}, Balance: {self.balance}"

    def is_active(self):
        return self.balance > 0

# Example usage
account1 = BankAccount("Khusbu", 5000)
print(account1.get_balance())
print(account1.deposit(2000))
print(account1.withdraw(1500))

if account1.is_active():
    print("The account is active.")
else:
    print("The account is inactive.")

class SavingsAccount(BankAccount):
    def __init__(self, owner, balance, interest_rate):
        super().__init__(owner, balance)
        self.interest_rate = interest_rate

    def add_interest(self):
        interest = self.balance * self.interest_rate / 100
        self.balance += interest
        return f"Interest added: {interest}. New balance: {self.balance}"

# Example usage
savings = SavingsAccount("Khusbu", 10000, 5)
print(savings.add_interest())
print(savings.get_balance())
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))
print("\n--- All Chunks ---")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
    print("-" * 50)