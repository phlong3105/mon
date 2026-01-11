# Useful Prompts

---

## Optimize

```text
Goal: Optimize code.

Act as a Senior Python Architect. Revise and refactor the provided code following these strict rules:

- ALWAYS refactor the code for performance, clean, readability, and production-ready. 
- Add comments to help maintainers understand the code's intetions.
- ALWAYS wrap lines at 100, can extend to 120 for readability. 
- ALWAYS align code by assignment(=, +=, -=, *=, /=), colon(:) and comma(,). Also support function call.

Check and optimize the code in $CURRENT_FILE
```

---

## Raise

```text
Goal: Revise "raise" statement ONLY. 

Act as a Senior Python Architect. Revise or generate raise statements for the provided code following these rules:
    
    1. EXAMPLES:
        - TypeError(f"Expected 'name' to be a string, but got {type(name).__name__}.")
        - ValueError(f"Expected 'split' in {valid_splits}, but got '{split}'")
        - ValueError(f"Unsupported 'split': {split}. Must be one of: {valid_splits}.")
        - AttributeError(f"'{type(self).__name__}' object has no attribute '_datapoints'.")
        - TypeError(f"Class {self.__name__} must define '_name' attribute.")
        - FileNotFoundError(f"Dataset root not found at: {path}")
        - FileExistsError(f"Export directory already exists: {path}")
        - IndexError(f"Index {index} out of range for dataset of size {len(self)}.")
        - KeyError(f"Model '{model_name}' not found in the model registry.")
        - NotImplementedError("This method is not yet supported.")
        - ImportError("Please install 'segment-anything' to use SAMSegmentor.")
        - RuntimeError("CUDA out of memory during SAM mask generation.")
        - For abstractmethod, just put "pass".
    
    2. FORMATS:
        - ALWAYS wrap lines at 100, can extend to 120 for readability. 
        - ONLY put variable name between single quote (e.g., 'name'). DO NOT put type between single quote.
        - Use: 
            - numpy.ndarray instead of np.ndarray.
            - str instead string.
    
 Please process with the following code: 
 $CURRENT_FILE
```

---

## Docstrings

```text
Goal: Revise docstrings ONLY.

Act as a Senior Python Architect. Revise or generate docstrings for the provided code following these strict structural and grammatical rules:
  
    1. General Rules:
        - Use Google-style docstrings, imperative-style.
        - Wrap lines at 80, can extend to 100 for readability. 
        - The first line must be a single, short summary (<79 chars) ending with a period.
    
    2. Package Docstrings:
        - The First line MUST be a descriptive Noun Phrase. Follow with ONE sentence of what the package contains, starting with "This package contains...". Nothing else. 
    
    3. Module Docstrings:
        - The First line MUST be a descriptive Noun Phrase. Follow with ONE sentence of what module's provides, starting with "This module provides...". Nothing else.
    
    4. Class Docstrings:
        - First line MUST be a Noun Phrase. The rest MUST use imperative-style.
        - Every attribute MUST include inline type hints (e.g., `data (numpy.ndarray): Description.`). Use full name instead of alias (e.g., numpy.ndarray instead of np.ndarray).
        - Omit the "Attributes:" section if there are no attributes.
        - Omit any "Args:" or "Returns:" sections.
        - When referring to an attribute or argument, put it between double quotes (e.g., ``name``). DO NOT PUT type between double quote (e.g., numpy.ndarray instead of ``numpy.ndarray``).

    5. Function/Method Docstrings:           
        - MUST use with an Imperative Verb (e.g., "Calculate...", "Resize...", "Return...").
        - "Args:" and "Returns:" sections MUST NOT contain inline type hints (rely on the code signature). Add "Defaults to ..." if a default value is given.
        - Omit the "Returns:" section entirely if the function only returns values (e.g., property or getter method).
        - Omit the "Returns:" section entirely if the function returns None.
        - Include a "Raises:" section for any explicitly raised exceptions.
        - When referring to an attribute or argument, put it between double quotes (e.g., ``name``). DO NOT PUT type between double quote (e.g., numpy.ndarray instead of ``numpy.ndarray``).
    
 Please process with the following code: 
 $CURRENT_FILE
```
