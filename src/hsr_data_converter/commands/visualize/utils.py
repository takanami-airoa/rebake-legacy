import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import ConnectionPatch


def draw_time_data(data, data_times, start_frame, end_frame, save_path="time_data.png"):
    """
    Plots the message arrival times for each topic within the specified frame range
    and displays `image_head` images at 0.1-second intervals at the top.

    Also plots the contents of `data_act_times` and `data_data_times` in red.

    Additionally, horizontal lines are drawn above all topics, and lines are drawn
    from the timestamp points of each topic to the intersection of those horizontal
    lines and the vertical lines from the original `time_stamp`.

    Parameters
    ----------
    data : dict
        A dictionary obtained from `extract_single_trajectory()`
        (uses `time_stamp` and `image_head` here).
    data_times : dict
        A dictionary where keys are action names and values are lists of
        message arrival times for those actions (original data).
    start_frame : int
        Start frame index to display (corresponds to `data["time_stamp"]`).
    end_frame : int
        End frame index to display.
    save_path : str
        Path to save the graph.
    """

    time_stamps = data.pop("time_stamp")

    # Apply Seaborn style
    # sns.set(style="whitegrid")

    # Determine the time range for visualization based on `data["time_stamp"]`
    start_time = time_stamps[start_frame]
    end_time = time_stamps[end_frame]

    length = len(time_stamps)
    width = int(length * 0.8)

    # Slice data
    for k in data.keys():
        data[k] = data[k][start_frame : end_frame + 1]

    # Configure the figure layout: images on top (1 row), timeline below (1 row)
    fig = plt.figure(figsize=(width, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.3)

    # Top section: Place `image_head` images
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_title("image_head", fontsize=12)
    ax_img.axis("off")

    # Use `time_stamp` and `image_head` as samples
    img_times = time_stamps
    images = data["image_head"]

    for i, img in enumerate(images):
        t = img_times[i]
        imagebox = OffsetImage(img, zoom=0.1)
        y = 0.7 if i % 2 == 0 else 0.0
        ab = AnnotationBbox(imagebox, (t, y), frameon=False, box_alignment=(0.5, 0.5))
        ax_img.add_artist(ab)
    ax_img.set_xlim([start_time, end_time])
    ax_img.set_ylim([0, 1])

    # Bottom section: Plot the timeline
    ax = fig.add_subplot(gs[1])
    ax.set_title("Topic Message Arrival Timeline", fontsize=14)
    ax.set_xlabel("Time [s]", fontsize=12)

    # Create a list of timestamps from original topics (plot in blue)
    topic_time_dict = {}
    for topic, t_list in data_times.items():
        filtered = sorted([t for t in t_list if start_time <= t <= end_time])
        if len(filtered) > 0:
            topic_time_dict[topic] = filtered

    # Additional data (`_time` in `data`) is plotted in red and organized into a separate dictionary
    topic_time_dict_red = {}
    for key in data.keys():
        if not key.endswith("time"):
            continue
        topic = re.match(r"(.*?)_time", key).group(1)
        topic_time_dict_red.setdefault(topic, []).extend(data[key])

    # Create a list of topics to display in order (image-related topics are displayed at the top)
    all_topics = set(list(topic_time_dict.keys()) + list(topic_time_dict_red.keys()))
    image_topics = [k for k in all_topics if "image" in k]
    non_image_topics = [k for k in all_topics if "image" not in k]
    topics_ordered = image_topics + sorted(non_image_topics)
    num_topics = len(topics_ordered)
    y_positions = {topic: num_topics - 1 - i for i, topic in enumerate(topics_ordered)}

    yticks = list(y_positions.values())
    ytick_labels = [
        topic for topic, _ in sorted(y_positions.items(), key=lambda x: -x[1])
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlim([start_time, end_time])
    ax.set_ylim([-1, num_topics])
    # Add horizontal lines along the y-coordinates of each topic
    for topic, y in y_positions.items():
        ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add new horizontal lines (x-direction above each topic)
    # Draw horizontal lines slightly above each topic's y-coordinate (e.g., 0.3 above)
    horizontal_offset = 0.5
    horizontal_lines = {}
    for topic, y in y_positions.items():
        y_line = y + horizontal_offset
        horizontal_lines[topic] = y_line
        ax.hlines(
            y=y_line,
            xmin=start_time,
            xmax=end_time,
            colors="gray",
            linestyles="--",
            linewidth=1,
            alpha=0.7,
        )

    # Plot original blue points (timestamps for each topic)
    for topic in topics_ordered:
        if topic in topic_time_dict:
            t_list = topic_time_dict[topic]
            y_val = np.full(len(t_list), y_positions[topic])
            ax.plot(
                t_list,
                y_val,
                linestyle="None",
                marker="o",
                markersize=5,
                color="blue",
                label=f"{topic} (orig)",
            )

    # Plot additional red points
    for topic in topics_ordered:
        if topic in topic_time_dict_red:
            t_list = topic_time_dict_red[topic]
            y_val = np.full(len(t_list), y_positions[topic])

            assert len(time_stamps) == len(t_list)
            ax.plot(
                t_list,
                y_val,
                linestyle="None",
                marker="o",
                markersize=5,
                color="red",
                label=f"{topic} (data)",
            )

    # Draw vertical lines for every 0.1 second (based on `time_stamp`)
    for t in time_stamps:
        ax.axvline(x=t, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Draw lines from each topic's timestamp points to the intersections of new horizontal lines and vertical lines
    for topic in topics_ordered:
        if topic in topic_time_dict_red:
            t_list = topic_time_dict_red[topic]
            for i, t in enumerate(t_list):
                t_real = time_stamps[i]
                con = ConnectionPatch(
                    xyA=(t_real, horizontal_lines[topic]),
                    coordsA=ax.transData,
                    xyB=(t, y_positions[topic]),
                    coordsB=ax.transData,
                    arrowstyle="-",
                    color="purple",
                    linewidth=0.5,
                    clip_on=True,
                )
                ax.add_artist(con)

    # Example: Connect `image_head` images with the timeline row labeled `image_head` (black lines)
    if "image_head" in y_positions and "image_head" in topic_time_dict_red:
        real_img_times = topic_time_dict_red["image_head"]
        for i, t in enumerate(img_times):
            if i >= len(real_img_times):
                break
            t_real = real_img_times[i]
            con = ConnectionPatch(
                xyA=(t_real, y_positions["image_head"]),
                coordsA=ax.transData,
                xyB=(t, 0.0),
                coordsB=ax_img.transData,
                arrowstyle="-",
                color="black",
                linewidth=0.5,
                clip_on=True,
            )
            fig.add_artist(con)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
