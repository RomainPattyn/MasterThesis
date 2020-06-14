def err1(alpha_g, alpha_f):
    return min((360 - alpha_f + alpha_g) % 180, (360 - alpha_g + alpha_f) % 180)


def err2(alpha_g, alpha_f):
    return abs((alpha_f - alpha_g + 180) % 360 - 180)


def err3(alpha_g, alpha_f):
    return abs((alpha_f - alpha_g + 360) % 360 - 180)


def print_err(alpha_g_list, alpha_f, err):
    for alpha_g in alpha_g_list:
        print(alpha_g, alpha_f, err(alpha_g, alpha_f))


alpha_g_list = list(range(0, 360, 20))
