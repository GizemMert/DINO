import lief

# Load the library
lib = lief.parse("/lib64/libldap.so.2")

# Find and remove the EVP_md2 symbol
sym = next((i for i in lib.imported_symbols if i.name == "EVP_md2"), None)
if sym:
    lib.remove_dynamic_symbol(sym)

# Save the modified library
lib.write("/home/aih/gizem.mert/tools/apps/mamba/envs/myenv/lib/libldap.so.2")
