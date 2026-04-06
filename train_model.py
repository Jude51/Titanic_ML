import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Твои пути
PATH1 = r'C:\Users\CAT\Downloads\archive\credit_risk_dataset.csv'
PATH2 = r'C:\Users\CAT\Downloads\GiveMeSomeCredit-training.csv'

def train():
    print("⏳ Загрузка и объединение данных...")
    
    # 1. Загружаем первый файл (Kaggle Credit Risk)
    df1 = pd.read_csv(PATH1)
    # Оставляем только нужные нам колонки и переименовываем для ясности
    df1 = df1[['person_age', 'person_income', 'loan_amnt', 'loan_status']]
    
    # 2. Загружаем второй файл (Give Me Some Credit)
    df2 = pd.read_csv(PATH2)
    # В этом файле колонки называются иначе. Маппим их:
    # age -> person_age, MonthlyIncome -> person_income (умножаем на 12 для годового), 
    # SeriousDlqin2yrs -> loan_status (инвертируем, т.к. там 1 - это просрочка)
    df2_cleaned = pd.DataFrame()
    df2_cleaned['person_age'] = df2['age']
    df2_cleaned['person_income'] = df2['MonthlyIncome'] * 12 # делаем годовой доход
    df2_cleaned['loan_amnt'] = df1['loan_amnt'].median() # во 2-м файле нет суммы, берем медиану из 1-го
    df2_cleaned['loan_status'] = df2['SeriousDlqin2yrs']
    
    # 3. Соединяем таблицы (Stacking)
    full_df = pd.concat([df1, df2_cleaned], axis=0, ignore_index=True)
    
    # Очистка пустых значений
    full_df = full_df.dropna()
    
    # Добавляем расчетные колонки, которые будут в боте
    full_df['loan_int_rate'] = 16.0
    full_df['loan_term_years'] = 5
    full_df['loan_percent_income'] = full_df['loan_amnt'] / full_df['person_income']
    
    # Удаляем бесконечные значения (если доход 0)
    full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna()

    # 4. Обучение
    features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_term_years', 'loan_percent_income']
    X = full_df[features]
    y = 1 - full_df['loan_status'] # 1 станет "Одобрено", 0 - "Отказ"

    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)

    # 5. Сохранение
    joblib.dump(model, 'credit_model.pkl')
    print(f"✅ Успех! Модель обучена на {len(full_df)} строках данных.")

if __name__ == "__main__":
    train()
