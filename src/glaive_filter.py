import json, jsonlines
from traceback import print_exc
from pprint import pprint



ROLE = {
    "user": "### User:",
    "assistant": "### Assistant:",
    "function-call": "### Function-Call:",
    "function-response": "### Function-Response:",
}
ROLE2ROLE = {
    "user_translated": "user",
    "사용자": "user",
    "assistant_translated": "assistant",
    "보조": "assistant",
    "비서": "assistant",
    "조수": "assistant",
    "어시스턴트": "assistant",
    "대화원": "assistant",
    "함수-호출": "function-call",
    "함수_호출": "function-call",
    "함수 호출": "function-call",
    "펑션 호출": "function-call",
    "기능-호출": "function-call",
    "함수-응답": "function-response",
    "함수_응답": "function-response",
    "함수 응답": "function-response",
    "기능-응답": "function-response",
    "펑션 응답": "function-response",

}

def join_for_mt(convs):
    lines = []
    for conv in convs:
        role = ROLE[conv["role"]]
        msg = conv.get("content") or conv["contenI"]

        if conv["role"] in ["user", "assistant"]:
            lines.append(f"{role}\n{msg}")

    return "\n\n".join(lines)
    

with jsonlines.open("../data/glaive-function-calling-v2-ko.json") as fin:
    items = list(fin)
    failed = 0 
    for item in items:
        try:
            ko = json.loads(item.pop("conversations-ko"))

            if isinstance(ko[0], list):
                ko = ko[0]

            item["conversations_ko"] = ko

            for conv in item["conversations_ko"]:
                conv["role"] = ROLE2ROLE.get(conv["role"]) or conv["role"]
        except:
            failed += 1

    items = [x for x in items if "conversations_ko" in x]
            
    print(f"{failed} items failed")


with jsonlines.open("../data/glaive-function-calling-v2-ko-mt/train.json", "w") as fmt,\
    jsonlines.open("../data/glaive-function-calling-v2-ko/train.json", "w") as ftrain:
    failed = 0 
    filtered_items = []
    for item in items:
        try:
            en = item.pop("conversations")
            ko = item.pop("conversations_ko")

            item["ko"] = join_for_mt(ko)
            item["en"] = join_for_mt(en)

            fmt.write(item)

            del item["ko"]
            del item["en"]

            item["conversations_ko"] = ko
            item["conversations"] = en
            ftrain.write(item)

        except json.JSONDecodeError:
            print_exc()
            print(item)
            failed += 1
        except:
            print_exc()
            print(item)
            pprint(ko)
            failed += 1


    print(f"{failed} items failed")