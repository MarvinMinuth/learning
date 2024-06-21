import matplotlib.pyplot as plt
from langchain_core.tools import tool


async def process_plot(plt, filename: str = "plot"):
    plt.savefig(filename, format="jpg")


@tool
async def create_pie_chart(data: dict, filename: str = "torte") -> None:
    """
    This function creates a pie chart from a given dictionary and saves it as a JPG file.

    :param data: Dictionary with names and normalized percentage values
    :param filename: Name of the JPG file to be saved
    """
    # Extracting data
    labels = list(data.keys())
    sizes = list(data.values())

    # Creating the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")  # Ensure equal aspect ratio

    # Saving as JPG
    plt.savefig(filename, format="jpg")
    plt.close()


@tool
async def create_bar_chart(data: dict, filename: str = "bar_chart") -> None:
    """
    This function creates a bar chart from a given dictionary and saves it as a JPG file.

    :param data: Dictionary with names and normalized percentage values
    :param filename: Name of the JPG file to be saved
    """
    # Extracting data
    labels = list(data.keys())
    values = list(data.values())

    # Creating the bar chart
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")
    ax.set_title("Bar Chart")

    # Saving as JPG
    process_plot(plt, filename=filename)
    plt.close()


@tool
async def create_line_graph(data: dict, filename: str = "line_graph") -> None:
    """
    This function creates a line graph from a given dictionary and saves it as a JPG file.

    :param data: Dictionary with names and normalized percentage values
    :param filename: Name of the JPG file to be saved
    """
    # Extracting data
    labels = list(data.keys())
    values = list(data.values())

    # Creating the line graph
    fig, ax = plt.subplots()
    ax.plot(labels, values, marker="o")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")
    ax.set_title("Line Graph")

    # Saving as JPG
    process_plot(plt, filename=filename)
    plt.close()
