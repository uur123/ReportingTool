
def structural_weakness_index(
    porosity,
    crack_density,
    avg_crack_length
):

    alpha = 0.5
    beta = 10000
    gamma = 0.05

    swi = (
        alpha * porosity
        +
        beta * crack_density
        +
        gamma * avg_crack_length
    )

    return round(swi, 3)
