import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. ПОДГОТОВКА ДАННЫХ ---
# Создадим синтетические данные для примера (чтобы код работал сразу)
# В реальности используйте свои пути к файлам
data = {
    'person_age': np.random.randint(20, 65, 1000),
    'person_income': np.random.randint(300000, 5000000, 1000),
    'loan_amnt': np.random.randint(50000, 2000000, 1000),
    'loan_int_rate': np.random.uniform(10, 25, 1000),
    'loan_status': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
    'loan_term_years': np.random.randint(1, 15, 1000)
}
df = pd.DataFrame(data)
df['loan_percent_income'] = (df['loan_amnt'] / 5) / df['person_income']

# Обучение
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- 2. ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС ---
print("\n--- ПРОФЕССИОНАЛЬНЫЙ БАНКОВСКИЙ СКОРИНГ ---")
try:
    age = int(input("Ваш возраст: "))
    income = float(input("Ваш годовой доход (руб): "))
    loan_request = float(input("Желаемая сумма кредита: "))
    years = int(input("Срок кредита (лет): "))
    rate = 16.0 # Текущая ставка
except ValueError:
    print("Ошибка ввода. Введите числа.")
    exit()

# Расчет аннуитета
m_rate = (rate / 100) / 12
months = years * 12
monthly_pay = loan_request * (m_rate * (1 + m_rate)**months) / ((1 + m_rate)**months - 1)
pti = monthly_pay / (income / 12)

# Подготовка данных для модели
user_case = pd.DataFrame([[age, income, loan_request, rate, years, (monthly_pay * 12 / income)]], 
                         columns=X.columns)
prob_success = model.predict_proba(user_case)[0][0]

# --- 3. АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ---
print(f"\nЕжемесячный платеж: {monthly_pay:,.2f} руб.")
print(f"Нагрузка на доход (PTI): {pti:.2%}")

decision = ""
reason = ""

if pti > 0.5:
    decision = "ОТКАЗ (Превышен лимит нагрузки)"
    max_payment = (income / 12) * 0.5
    # Обратный расчет суммы: P = PMT / ( (r*(1+r)^n) / ((1+r)^n - 1) )
    max_loan = max_payment / ((m_rate * (1 + m_rate)**months) / ((1 + m_rate)**months - 1))
    reason = f"Платеж превышает 50% вашего дохода. Мы можем предложить не более {max_loan:,.0f} руб."
elif age + years > 70:
    decision = "ОТКАЗ (Возрастной ценз)"
    reason = "Срок кредита выходит за рамки пенсионного возраста (70 лет)."
elif prob_success < 0.6:
    decision = "ОТКАЗ (Риск системы)"
    reason = "Модель оценивает ваш профиль как высокорисковый на основе статистических данных."
else:
    decision = "ОДОБРЕНО"
    reason = "Ваш доход и возраст соответствуют политике банка. Риски минимальны."

print(f"\nВЕРДИКТ: {decision}")
print(f"ОБОСНОВАНИЕ: {reason}")

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(12, 5))

# График 1: Важность факторов
plt.subplot(1, 2, 1)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importances.plot(kind='barh', color='skyblue')
plt.title("На что смотрел банк (Важность факторов)")

# График 2: Шанс одобрения
plt.subplot(1, 2, 2)
colors = ['red', 'yellow', 'green']
plt.bar(['Шанс одобрения'], [prob_success * 100], color='green' if prob_success > 0.6 else 'red')
plt.ylim(0, 100)
plt.axhline(60, color='black', linestyle='--', label='Порог прохода')
plt.title(f"Скоринг-балл: {prob_success*100:.1f}%")
plt.legend()

plt.tight_layout()
plt.show()
