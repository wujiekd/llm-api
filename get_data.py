import argparse
import json
import requests
import json
import yaml


def main(args):
    """
    Call LLM API
    """
    headers = {
        'Content-Type': 'application/json',
    }
    with open(args.config_fn, "r") as reader:
        config = yaml.safe_load(reader)
        if any(k not in config for k in ["template", "options"]):
            raise KeyError("config file must have the following keys: "
                           f"'template' and 'option'!")
        template = config["template"]
        options = config["options"]
        print(json.dumps(config, indent=2, ensure_ascii=False))

    # read test sample
    dialogs = []
    with open(args.input_fn, "r") as reader:
        for line in reader.readlines():
            input = json.loads(line.strip())
            dialogs.append(input["question"])
    
    # run batch prediction
    with open(args.output_fn, "w") as writer:
        for i, dialog in enumerate(dialogs):
            prompt = template.format(dialog, options)
            messages = {"prompt": prompt}
            response = requests.post(args.url, headers=headers, 
                                     data=json.dumps(messages))
            if response.status_code != 200:
                print(f"Something is wrong: {response.status_code}")
            
            # default output
            output = {'input': messages}
            try:
                output.update(json.loads(response.content))
            except Exception as e:
                print(str(e))
                output["status"] = response.status_code

            writer.write(json.dumps(output, 
                                    ensure_ascii=False) + '\n')
            print(i, prompt)
            print(f"response: {output['response']}\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM prediction")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--config_fn", type=str, required=True)
    parser.add_argument("--input_fn", type=str, required=True)
    parser.add_argument("--output_fn", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
    


