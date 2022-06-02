from datetime import timedelta
import pandas as pd

products_path = "data/products.jsonl"
sessions_path = "data/sessions.jsonl"
users_path = "data/users.jsonl"

products_data = pd.read_json(products_path, lines=True)
sessions_data = pd.read_json(sessions_path, lines=True)
users_data = pd.read_json(users_path, lines=True)
sessions_data = pd.merge(sessions_data, products_data, on="product_id").drop(
        columns=['product_name', 'category_path']).sort_values(by=['timestamp'])


def main():
    prev_purchase = {}
    spending_factors = {}
    buy_session_counter = {}

    for user_id in users_data["user_id"]:
        prev_purchase[user_id] = None
        spending_factors[user_id] = 0
        buy_session_counter[user_id] = 0

    for session in sessions_data.iterrows():
        session = session[1]
        user_id = session["user_id"]
        if session["event_type"] == "VIEW_PRODUCT":
            continue
        buy_session_counter[user_id] += 1
        if prev_purchase[user_id]:
            time_difference = max(session["timestamp"] - prev_purchase[user_id], timedelta(hours=24)) / 3600
            time_difference = time_difference.total_seconds()
            spending_factors[user_id] += session["price"] * (1-session["offered_discount"]/100) / time_difference
        prev_purchase[user_id] = session["timestamp"]

    for user_id in spending_factors.keys():
        if not buy_session_counter[user_id]:
            spending_factors[user_id] = 0.0
        else:
            spending_factors[user_id] /= buy_session_counter[user_id]
        print(user_id, ": ", spending_factors[user_id])

    return spending_factors


if __name__ == "__main__":
    print(main())
