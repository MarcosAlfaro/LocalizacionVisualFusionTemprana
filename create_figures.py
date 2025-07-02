"""
THIS PROGRAM CONTAINS FUNCTIONS THAT CREATE FIGURES THAT GRAPHICALLY REPRESENT THE TEST RESULTS OF A LOCALIZATION STAGE
Among these graphics, we can find:
- Confusion matrices (coarse step)
- Maps with the network predictions (fine and global steps)
- Graphics that show the errors made by the network in every room (fine step)
Test programmes will call these functions
"""

# Set the 'Agg' backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require a display server

import matplotlib.pyplot as plt
import os


def display_coord_map(direc, mapVM, mapTest, k, ilum, imgFormat):
    # Cargar la fuente personalizada
    # times_new_roman = font_manager.FontProperties(fname=font_path)

    # Usar la fuente cargada en la figura
    # plt.rcParams["font.family"] = times_new_roman.get_name()
    plt.rcParams["xtick.labelsize"] = 16  # Tamaño de fuente para el eje X
    plt.rcParams["ytick.labelsize"] = 16  # Tamaño de fuente para el eje Y

    xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000
    plt.figure(figsize=(9, 6), dpi=120, edgecolor='black')

    firstk1, firstErrork, firstErrorRoom = True, True, True

    for vm in range(len(mapVM)):
        if vm == 0:
            plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue', s=30)
            plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue', s=30, label="Modelo Visual")
        else:
            plt.scatter(mapVM[vm][0], mapVM[vm][1], s=30, color='blue')

    for t in range(len(mapTest)):
        if mapTest[t][4] == "R":
            if firstErrorRoom:
                plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='brown')
                firstErrorRoom = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='brown')
        elif mapTest[t][4] == "F":
            if firstErrork:
                plt.scatter(mapTest[t][2], mapTest[t][3], s=30, color='red', label="Incorrect Prediction at recall 20")
                plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='red')
                firstErrork = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='red')
        else:
            label = int(mapTest[t][4])
            if label < k/2:
                color = (2*label/k, 1, 0)
            elif label > k/2:
                color = (1, 1 - 2*(label - k/2) / k, 0)
            else:
                color = (1, 1, 0)
            if firstk1:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='green', s=25)
                plt.scatter(mapTest[t][2], mapTest[t][3], color='green', s=30, label='Imágenes Test')
                firstk1 = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], s=30, color=color)

    plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
    plt.ylabel('y (m)', fontsize=18)
    plt.xlabel('x (m)', fontsize=18)
    plt.title(f'Entorno: {imgFormat}, Iluminación: {ilum}', fontsize=22)
    plt.legend(fontsize=18, facecolor='white', edgecolor='black')
    plt.autoscale()
    plt.grid(lw=1.5)
    plt.savefig(os.path.join(direc, "map_" + imgFormat + "_" + ilum + ".png"), dpi=400)
    plt.close()

    return




def display_confidence_map(direc, mapTest, features, bestConfFeatures, label, env, ilum):
    """
    Generates and saves a scatter plot confidence map with styling adapted
    from the reference bar chart script.
    """
    # Create the plot and the axes object using the object-oriented approach
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')

    # --- Custom Color & Marker Palette (from user's function) ---
    colors = ["#3092CF", "#ED6D1D", "#58A525", "#F64B48", "#F26FE7"]
    markers = ["x", "."]

    # --- Standardized Fontsize Settings (from reference script) ---
    TITLE_FONTSIZE = 32
    LABEL_FONTSIZE = 26 # Kept the user's original larger size for axis labels
    TICK_FONTSIZE = 22
    LEGEND_FONTSIZE = 24 # Kept the user's original larger size for the legend

    # This logic handles labeling each feature type only once for the legend
    firstElements = {
        "Color": True,
        "Intensity": True,
        "Gradient (Mag.)": True,
        "Gradient (Theta)": True,
        "Hue": True
    }
    
    feature_map = {
        "COLOR": ("Color", colors[0]),
        "GRAYSCALE": ("Intensity", colors[1]),
        "MAGNITUDE": ("Gradient (Mag.)", colors[2]),
        "ANGLE": ("Gradient (Theta)", colors[3]),
        "HUE": ("Hue", colors[4]) # Assuming HUE maps to the 5th color
    }
    # Add a fallback for features not in the map
    if "HUE" not in features:
        features.append("HUE")

    for t in range(len(mapTest)):
        # This condition seems to be for sampling the data
        # if t % 3 != 0:
        #     continue

        marker = markers[label[t]]
        feature_name = bestConfFeatures[t]

        # Determine the label and color for the current point
        if feature_name in feature_map:
            current_label, current_color = feature_map[feature_name]
        else:
            # Fallback for any other features
            idx = features.index(feature_name) + 1
            current_label = feature_name
            current_color = colors[idx % len(colors)]


        # Plot the point, adding a label only for the first instance of each type
        if firstElements.get(current_label, False):
            ax.scatter(mapTest[t][0], mapTest[t][1], s=50, color=current_color, marker=marker, label=current_label)
            firstElements[current_label] = False
        else:
            ax.scatter(mapTest[t][0], mapTest[t][1], s=50, color=current_color, marker=marker)

    # Apply styling consistent with the reference script
    ax.autoscale()
    ax.set_ylabel('y (m)', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel('x (m)', fontsize=LABEL_FONTSIZE)
    ax.set_title(f'Confidence map: 360Loc {env} {ilum}', fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, facecolor='white', edgecolor='black')

    # Set tick font sizes
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    # Use the grid style from the reference script
    ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)

    # Save the figure using the axes' figure object
    fig.savefig(os.path.join(direc, f"map_{env}_{ilum}.png"), dpi=400)
    plt.close(fig) # Close the specific figure to free memory

    return


# def display_coord_map(direc, mapVM, mapTest, k, ilum, imgFormat):
#     # Cargar la fuente personalizada
#     # times_new_roman = font_manager.FontProperties(fname=font_path)
#
#     # Usar la fuente cargada en la figura
#     # plt.rcParams["font.family"] = times_new_roman.get_name()
#     plt.rcParams["xtick.labelsize"] = 16  # Tamaño de fuente para el eje X
#     plt.rcParams["ytick.labelsize"] = 16  # Tamaño de fuente para el eje Y
#
#     xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000
#     plt.figure(figsize=(9, 6), dpi=120, edgecolor='black')
#
#     firstk1, firstErrork, firstErrorRoom = True, True, True
#
#     for vm in range(len(mapVM)):
#         if vm == 0:
#             plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue', s=30)
#             plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue', s=30, label="Modelo Visual")
#         else:
#             plt.scatter(mapVM[vm][0], mapVM[vm][1], s=30, color='blue')
#         xmax, xmin, ymax, ymin = get_axes_limits(mapVM[vm][0], mapVM[vm][1], xmax, xmin, ymax, ymin)
#
#     for t in range(len(mapTest)):
#         if mapTest[t][4] == "R":
#             if firstErrorRoom:
#                 plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='brown')
#                 # plt.scatter(mapTest[t][2], mapTest[t][3], s=30, color='brown', label="Wrong Room")
#                 firstErrorRoom = False
#             else:
#                 plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='brown')
#                 plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], 'brown', lw=1)
#             xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
#         elif mapTest[t][4] == "F":
#             if firstErrork:
#                 plt.scatter(mapTest[t][2], mapTest[t][3], s=30, color='red', label="Predicción incorrecta en recall@20")
#                 plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='red')
#                 firstErrork = False
#             else:
#                 plt.scatter(mapTest[t][2], mapTest[t][3], s=25, color='red')
#                 plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], 'red', lw=1)
#             xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
#         else:
#             label = int(mapTest[t][4])
#             if label < k/2:
#                 color = (2*label/k, 1, 0)
#             elif label > k/2:
#                 color = (1, 1 - 2*(label - k/2) / k, 0)
#             else:
#                 color = (1, 1, 0)
#             if firstk1:
#                 plt.scatter(mapTest[t][2], mapTest[t][3], color='green', s=25)
#                 plt.scatter(mapTest[t][2], mapTest[t][3], color='green', s=30, label='Predicción correcta en recall@1')
#                 firstk1 = False
#             else:
#                 plt.scatter(mapTest[t][2], mapTest[t][3], s=30, color=color)
#                 plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color=color, lw=1)
#             xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
#
#     plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
#     plt.ylabel('y (m)', fontsize=18)
#     plt.xlabel('x (m)', fontsize=18)
#     plt.title(f'Entorno: {imgFormat}, Iluminación: {ilum}', fontsize=22)
#     # plt.legend(fontsize=13, facecolor='white', edgecolor='black')
#     plt.grid(lw=1.5)
#     plt.savefig(os.path.join(direc, "mapa_" + imgFormat + "_" + ilum + ".png"), dpi=400)
#     # plt.savefig(os.path.join(direc, "leyenda_traducida.png"), dpi=400)
#     plt.close()
#
#     return
