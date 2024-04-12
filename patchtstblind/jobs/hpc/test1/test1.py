import argparse

def main():
    from rich.console import Console
    # Check if an argument is passed
    parse = argparse.ArgumentParser(description="test argument parser")
    parse.add_argument("-n", "--N", type=int, default=1, required=True, help="take a random integer value")
    args = parse.parse_args()
    x = args.N

    # Perform a simple operation, like squaring the number
    result = x ** 2

    # Print the results
    # console = Console()
    # console.print(f"Received number: {x}")
    # console.print(f"Square of the number: {result}")

    print(f"Received number: {x}")
    print(f"Square of the number: {result}")

if __name__ == "__main__":
    main()
