import itertools


def quinella_combination(
    race_id,
    favorite_horse_numbers,
    rival_horse_numbers,
    first_probas,
    second_probas,
    with_proba=False,
):
    """
    馬連の組み合わせを計算する

    Parameters
    ----------
    race_id : int
        レースID
    favorite_horse_nos : array
        軸馬の馬番
    rival_horse_numbers : array
        紐馬の馬番
    first_probas : array
        馬番順にソートされた1着確率
    second_probas : array
        馬番順にソートされた2着確率
    with_proba : Bool
        結果に勝利確率を含めるかどうか。計算する分遅くなる

    Returns
    -------
    df : pandas.DataFrame
        馬番の組み合わせを保持したデータフレーム。
    """

    comb_set = set()
    comb_array = []
    for comb in itertools.product(favorite_horse_numbers, rival_horse_numbers):
        if comb[0] == comb[1]:
            continue

        sorted_comb = sorted(comb)
        comb_str = f"{sorted_comb[0]}:{sorted_comb[1]}"

        if comb_str in comb_set:
            continue

        comb_set.add(comb_str)

        if with_proba:
            # horse_no1 -> horse_no2の確率
            proba1 = calc_quinella_proba(
                round(first_probas[sorted_comb[0] - 1], 4),
                round(second_probas[sorted_comb[0] - 1], 4),
                round(second_probas[sorted_comb[1] - 1], 4),
            )
            # horse_no2 -> horse_no1の確率
            proba2 = calc_quinella_proba(
                round(first_probas[sorted_comb[1] - 1], 4),
                round(second_probas[sorted_comb[1] - 1], 4),
                round(second_probas[sorted_comb[0] - 1], 4),
            )
            # 馬連なのでどちらでもよい
            proba = proba1 + proba2
            comb_array.append([race_id, sorted_comb[0], sorted_comb[1], proba])
        else:
            comb_array.append([race_id, sorted_comb[0], sorted_comb[1]])

    return comb_array


def calc_quinella_proba(p1_win_proba, p1_second_proba, p2_second_proba):
    """
    1着馬の確率と2着馬の確率を掛け合わせて1-2着確率を求める
    2着馬の確率は、1着馬を除いて再計算するため(1 - p2_predict)となる

    Parameters
    ----------
    p1_win_proba : float
        軸馬の1着確率
    p1_second_proba : float
        軸馬の2着確率
    p2_second_proba : float
        紐馬の2着確率

    Returns
    -------
    proba : float
        馬連確率
    """
    return (p1_win_proba * p2_second_proba) / (1 - p1_second_proba)
