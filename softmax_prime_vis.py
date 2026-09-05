import numpy as np
import math
import matplotlib.pyplot as plt

def inverse_softmax(outputs, target_denominator):
    soft_inputs = np.zeros(len(outputs))
    for i in range(len(outputs)):
        soft_inputs[i] = math.log(target_denominator * outputs[i])
    return soft_inputs

def cost_function(real_values, softmax_outputs):
    cost = np.zeros(len(real_values))
    for i, v in enumerate(zip(real_values, softmax_outputs)):
        cost[i] = (v[0] - v[1])**2
    return cost

def cost_prime(real_values, softmax_outputs):
    cost_deriv = np.zeros(len(real_values))
    for i, v in enumerate(zip(real_values, softmax_outputs)):
        cost_deriv[i] = 2 * (v[1] - v[0])
    return cost_deriv

def softmax(input_vec):
    exs = np.array([math.exp(i) for i in input_vec])
    denom = sum(exs)
    return exs * 1/denom

def softmax_prime(soft_outputs):
    output = np.zeros((len(soft_outputs), len(soft_outputs)))
    for i in range(len(soft_outputs)):
        for j in range(len(soft_outputs)):
            if i == j:
                output[i][j] = soft_outputs[i] * (1 - soft_outputs[i])
            else:
                output[i][j] = -soft_outputs[i] * soft_outputs[j]
    return output

def plot(real_values, step, N):
    input_vec = np.zeros(len(real_values))
    fig, axes = plt.subplots(len(input_vec), 2, figsize=(10, 4))
    for variable_idx in range(len(input_vec)):
        ax1, ax2 = axes[variable_idx][0], axes[variable_idx][1]
        x_axis = []

        y_one = []
        y_two = []
        for i in range(N):
            
            x_axis.append(input_vec[variable_idx] + step * i)
            new_input_vec = np.array(input_vec)
            new_input_vec[variable_idx] = new_input_vec[variable_idx] + step * i
            sfm = softmax(new_input_vec)
            c = sum(cost_function(real_values, sfm))
            y_one.append(c)

            sp = softmax_prime(sfm)
            cp = cost_prime(real_values, sfm)
            y_two.append((sp @ cp)[variable_idx])

        ax1.plot(x_axis, y_one, color="blue")
        ax1.set_title("cost")
        ax1.set_xlabel(f"input_vec_{variable_idx}")
        ax1.set_ylabel("cost")

        ax2.plot(x_axis, y_two, color="red")
        ax2.set_title("cost derivative")
        ax2.set_xlabel(f"input_vec_{variable_idx}")
        ax2.set_ylabel("cost'")







        

plots = plot([0.33, 0.33, 0.33], 0.005, 10000)
plt.tight_layout()
plt.show()