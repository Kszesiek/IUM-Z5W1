import math
from datetime import timedelta, datetime
import pandas as pd

deliveries_path = "./data/deliveries.jsonl"
products_path = "./data/products.jsonl"
sessions_path = "./data/sessions.jsonl"
users_path = "./data/users.jsonl"

deliveries_data = pd.read_json(deliveries_path, lines=True)
products_data = pd.read_json(products_path, lines=True)
sessions_data = pd.read_json(sessions_path, lines=True)
users_data = pd.read_json(users_path, lines=True)

sessions_data = pd.merge(sessions_data, products_data, on="product_id").drop(
        columns=['product_name', 'category_path']).sort_values(by=['timestamp'], ascending=False)

sessions_learn = sessions_data
sessions_validate = sessions_data

sessions_learn = sessions_learn[sessions_learn["timestamp"].dt.month != 5]
sessions_validate = sessions_validate[sessions_validate["timestamp"].dt.month == 5]


def predict(data):
    return None


def learn():
    next_purchase = {user_id: datetime.now() for user_id in users_data["user_id"]}
    spending_factors = {user_id: 0 for user_id in users_data["user_id"]}
    buy_session_counter = {user_id: 0 for user_id in users_data["user_id"]}
    actual_total = 0

    learning_days = (max(sessions_learn["timestamp"]) - min(sessions_learn["timestamp"])).days

    for session in sessions_learn.iterrows():
        session = session[1]
        user_id = session["user_id"]
        if session["event_type"] == "VIEW_PRODUCT":
            continue
        buy_session_counter[user_id] += 1
        actual_total += session["price"] * (1 - session["offered_discount"] / 100)
        if next_purchase[user_id]:
            time_difference = max(next_purchase[user_id] - session["timestamp"], timedelta(hours=24))
            time_difference = time_difference.total_seconds() / 86400
            days_from_today = (datetime.now() - session["timestamp"]).total_seconds() / 86400 / 30
            favor_newer_records_with_tanh = (0.5 + 0.5 * math.tanh(- days_from_today / 2 + 2))
            spending_factors[user_id] += session["price"] * (1 - session["offered_discount"] / 100) / time_difference * favor_newer_records_with_tanh
        next_purchase[user_id] = session["timestamp"]

    correction = actual_total / sum(spending_factors.values())
    spending_factors = {user_id: factor * correction for (user_id, factor) in spending_factors.items()}

    max_factor = max(spending_factors.values())
    spending_factors_perc = {key: val / max_factor * 100 for (key, val) in spending_factors.items()}
    monthly_est = {key: val * 30 / learning_days for (key, val) in spending_factors.items()}

    return monthly_est


def validate(model):
    total_for_user = {user_id: 0 for user_id in users_data["user_id"]}

    for session in sessions_validate.iterrows():
        session = session[1]
        user_id = session["user_id"]
        if session["event_type"] == "VIEW_PRODUCT":
            continue
        total_for_user[user_id] += session["price"] * (1 - session["offered_discount"] / 100)

    total = 0
    sum_ratio = 0
    for user_id in users_data["user_id"]:
        if total_for_user[user_id] == 0:
            ratio = 0 if model[user_id] < 10 else min(model[user_id] / 10, 1)
        else:
            ratio = min((model[user_id] - total_for_user[user_id]) / total_for_user[user_id], 1)
        print(f"user {user_id}: {ratio:3.2f} ({model[user_id]:3.1f}/{total_for_user[user_id]:3.1f})")
        total += total_for_user[user_id]
        sum_ratio += abs(ratio)
    print(f"dokładność: {100 * ( 1 - sum_ratio / len(users_data.index)):6.2f}%")
    print(f"suma modelu: {sum(model.values()):9.2f}")
    print(f"suma actual: {total:9.2f}")


if __name__ == "__main__":
    model = learn()
    validate(model)
