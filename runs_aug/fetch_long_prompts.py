import urllib.request
import json
import re
import os
import yaml

def fetch_wikipedia_plain_text(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=max&explaintext=1&titles={title}&format=json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if "extract" in page_data:
                    return page_data["extract"]
    except Exception as e:
        print(f"Error fetching {title}: {e}")
    return None

def clean_text(text):
    # Remove empty lines, excessive spaces, wikipedia section headers like === Section ===
    text = re.sub(r'={2,}\s*[^=]+\s*={2,}', '', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    articles = [
        "Artificial_intelligence",
        "World_War_II",
        "United_States",
        "Albert_Einstein",
        "Isaac_Newton"
    ]
    
    prompts = []
    print("Fetching articles from Wikipedia...")
    for title in articles:
        print(f"Fetching {title}...")
        raw_text = fetch_wikipedia_plain_text(title)
        if not raw_text:
            print(f"Failed to fetch {title}")
            continue
        cleaned = clean_text(raw_text)
        words = cleaned.split()
        # Keep first 12000 words to ensure at least 10k tokens (typically 1.3 - 1.4 tokens per word)
        truncated_words = words[:3000]
        final_text = " ".join(truncated_words)
        print(f"Retrieved {len(truncated_words)} words for {title} (character length: {len(final_text)})")
        prompts.append(final_text)
        
    if len(prompts) < 5:
        print(f"Error: Only fetched {len(prompts)} prompts. Need 5. Aborting.")
        return
        
    config_path = "/home/grego/Documents/BME/Thesis/TDK_LLM/runs_aug/jacobian_config.yaml"
    with open(config_path, "r") as f:
        config_content = f.read()
        
    # We want to format the prompts block nicely as YAML.
    # We write a custom formatter to generate the exact YAML format:
    #     - id: "internet_long_prompts"
    #       texts:
    #         - >
    #           text line 1...
    #           text line 2...
    
    yaml_lines = []
    yaml_lines.append('    - id: "internet_long_prompts"')
    yaml_lines.append('      texts:')
    
    for text in prompts:
        yaml_lines.append('        - >')
        # Wrap text at 80 characters for readability in YAML
        words_list = text.split()
        current_line = []
        current_len = 0
        for w in words_list:
            if current_len + len(w) + 1 > 80:
                yaml_lines.append('          ' + ' '.join(current_line))
                current_line = [w]
                current_len = len(w)
            else:
                current_line.append(w)
                current_len += len(w) + 1
        if current_line:
            yaml_lines.append('          ' + ' '.join(current_line))
            
    new_prompts_yaml = "\n".join(yaml_lines) + "\n"
    
    # We find where prompts list is in the config
    # In jacobian_config.yaml, we have prompts: followed by - id: "long_prompts"
    # We want to insert the new prompts block under prompts: and right after the last text list item.
    # Let's search for "  setups:" to insert just before it
    
    match = re.search(r'(\n\s*setups:)', config_content)
    if not match:
        print("Could not find setups: block in yaml to insert prompts.")
        return
        
    insert_pos = match.start()
    
    # Insert new prompts block before setups:
    # Ensure proper spacing
    updated_content = config_content[:insert_pos] + "\n" + new_prompts_yaml + "\n" + config_content[insert_pos:]
    
    with open(config_path, "w") as f:
        f.write(updated_content)
        
    print("Config successfully updated.")
    
    # Verification: Try parsing the file using yaml library to make sure it's valid
    try:
        with open(config_path, "r") as f:
            yaml.safe_load(f)
        print("YAML syntax verification PASSED.")
    except Exception as e:
        print(f"YAML syntax verification FAILED: {e}")

if __name__ == "__main__":
    main()
