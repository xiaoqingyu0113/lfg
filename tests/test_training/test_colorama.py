import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init(autoreset=True)

# Foreground colors
print(Fore.RED + 'This is red text')
print(Fore.GREEN + 'This is green text')
print(Fore.BLUE + 'This is blue text')

# Background colors
print(Back.YELLOW + 'This is text with a yellow background')
print(Back.CYAN + 'This is text with a cyan background')

# Style
print(Style.DIM + 'This is dim text')
print(Style.NORMAL + 'This is normal text')
print(Style.BRIGHT + 'This is bright text')

# Combining styles
print(Back.YELLOW + Fore.RED + 'This is red text with a yellow background')
print(Style.BRIGHT + Fore.GREEN + 'This is bright green text')

# Reset to default color
print('This is default text color')
