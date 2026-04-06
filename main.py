import asyncio
import io
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, BufferedInputFile

# --- НАСТРОЙКИ ---
TOKEN = "8770570629:AAGcqrfdFuR0lxr0eIZjT4Nb5di-RBU8rsg"
plt.rcParams['font.family'] = 'sans-serif' # Поддержка кириллицы

# --- 1. ЗАГРУЗКА МОДЕЛИ ---
try:
    model = joblib.load('credit_model.pkl')
    # Порядок колонок должен строго совпадать с порядком при обучении
    feature_names = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_term_years', 'loan_percent_income']
    print("✅ Модель успешно загружена")
except Exception as e:
    print(f"❌ ОШИБКА: Не удалось загрузить 'credit_model.pkl'. Сначала обучи модель!\n{e}")
    exit()

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Состояния опроса
class CreditForm(StatesGroup):
    age = State()
    income = State()
    amount = State()
    years = State()

def main_menu():
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="💳 Кредит")],
        [KeyboardButton(text="❤️ Здоровье"), KeyboardButton(text="📊 Отток")],
        [KeyboardButton(text="🏠 Цена квартиры")]
    ], resize_keyboard=True)

# --- 2. ОБРАБОТКА КОМАНД ---

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Я MLmao 🤖\nВыбери модель для проверки:", reply_markup=main_menu())

@dp.message(F.text == "💳 Кредит")
async def credit_start(message: types.Message, state: FSMContext):
    await message.answer("Введите ваш возраст (полных лет):")
    await state.set_state(CreditForm.age)

@dp.message(CreditForm.age)
async def process_age(message: types.Message, state: FSMContext):
    if not message.text.isdigit():
        return await message.answer("Введите возраст числом.")
    await state.update_data(age=int(message.text))
    await message.answer("Ваш годовой доход (в рублях):")
    await state.set_state(CreditForm.income)

@dp.message(CreditForm.income)
async def process_income(message: types.Message, state: FSMContext):
    val = message.text.replace(" ", "")
    try:
        await state.update_data(income=float(val))
        await message.answer("Желаемая сумма кредита:")
        await state.set_state(CreditForm.amount)
    except:
        await message.answer("Введите сумму цифрами.")

@dp.message(CreditForm.amount)
async def process_amount(message: types.Message, state: FSMContext):
    val = message.text.replace(" ", "")
    try:
        await state.update_data(amount=float(val))
        await message.answer("Срок кредита (в годах):")
        await state.set_state(CreditForm.years)
    except:
        await message.answer("Введите сумму цифрами.")

@dp.message(CreditForm.years)
async def process_final(message: types.Message, state: FSMContext):
    if not message.text.isdigit():
        return await message.answer("Введите срок числом.")
    
    data = await state.get_data()
    age = data['age']
    income = data['income']
    loan_req = data['amount']
    years = int(message.text)
    rate = 16.0 # Ставка по умолчанию

    # 1. Математические расчеты
    m_rate = (rate / 100) / 12
    months = years * 12
    annuity_factor = (m_rate * (1 + m_rate)**months) / ((1 + m_rate)**months - 1)
    monthly_pay = loan_req * annuity_factor
    pti = monthly_pay / (income / 12)
    loan_pct = loan_req / income # Доля кредита от годового дохода

    # 2. Предикт модели
    user_case = pd.DataFrame([[age, income, loan_req, rate, years, loan_pct]], columns=feature_names)
    prob_success = model.predict_proba(user_case)[0][1] # Вероятность одобрения

    # 3. Логика решения
    decision, reason, offer = "", "", ""
    safe_pay = (income / 12) * 0.35 # Безопасный платеж (35% дохода)
    guaranteed_loan = safe_pay / annuity_factor

    if pti > 0.5:
        decision = "❌ ОТКЛОНЕНО"
        reason = f"Слишком высокая финансовая нагрузка ({pti:.1%})."
        offer = f"💡 Попробуйте сумму до **{guaranteed_loan:,.0f} ₽**."
    elif age + years > 70:
        decision = "❌ ОТКЛОНЕНО"
        reason = f"Срок кредита превышает возрастной лимит (70 лет). Вам будет {age + years}."
        offer = f"💡 Попробуйте срок до **{70 - age} лет**."
    elif prob_success < 0.55:
        decision = "❌ ОТКЛОНЕНО"
        reason = "Система скоринга оценивает риски как высокие."
        offer = f"💡 Мы рекомендуем запросить **{guaranteed_loan:,.0f} ₽** для повышения шансов."
    else:
        decision = "✅ ОДОБРЕНО"
        reason = f"Ваш профиль надежен. Нагрузка {pti:.1%} в норме."
        offer = "✨ Вы можете прийти в банк за получением средств!"

    # 4. Визуализация
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    labels = {'person_age':'Возраст','person_income':'Доход','loan_amnt':'Сумма','loan_int_rate':'Ставка','loan_term_years':'Срок','loan_percent_income':'Нагрузка'}
    feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    feat_imp.index = [labels.get(x, x) for x in feat_imp.index]
    feat_imp.plot(kind='barh', color='skyblue')
    plt.title("На что смотрел ИИ")

    plt.subplot(1, 2, 2)
    plt.bar(['Шанс'], [prob_success * 100], color='green' if prob_success >= 0.55 else 'red')
    plt.ylim(0, 100)
    plt.axhline(55, color='black', linestyle='--')
    plt.title(f"Скоринг: {prob_success*100:.1f}%")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # 5. Отправка
    photo = BufferedInputFile(buf.read(), filename="result.png")
    caption = (f"**{decision}**\n\n**Причина:** {reason}\n\n{offer}\n\n"
               f"📉 Платеж: {monthly_pay:,.0f} ₽/мес\n"
               f"📊 Нагрузка (PTI): {pti:.1%}")

    await message.answer_photo(photo=photo, caption=caption, parse_mode="Markdown", reply_markup=main_menu())
    await state.clear()

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
