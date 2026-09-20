from std.python import Python, PythonObject


def main() raises:
    var typesafe = Python.import_module("typesafe_sdk")
    var json = Python.import_module("json")

    # api_key is required by Config.resolve() even though this local server does
    # not check it; base_url points at the my_jev server, not the real TypeSafe API.
    var client = typesafe.TypeSafeClient(
        api_key="local",
        base_url="http://localhost:8000",
        model="gemma3:4b-it-qat",
    )

    var state = "I was charged twice for the same order. Please help ASAP, the server is on fire."

    # Built via json.loads rather than nested Mojo dict literals: PythonObject dict
    # construction is generic over one value type, which cannot express this
    # heterogeneous, nested question payload.
    var questions_json = """
    {
        "billing": {
            "type": "noul",
            "instructions": "Is this message about a billing problem?",
            "criteria": {
                "true": "The message concerns a billing or charge issue.",
                "false": "The message is unrelated to billing."
            }
        },
        "tone": {
            "type": "choice",
            "instructions": "What is the tone of this message?",
            "criteria": {
                "angry": "An upset or hostile message",
                "calm": "A neutral or polite message",
                "excited": "An enthusiastic or eager message"
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this message?",
            "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
        }
    }
    """
    var questions = json.loads(questions_json)

    var result = client.system_one(state, questions)

    print("model:", String(py=result.model))

    var billing = result.nouls["billing"]
    print("billing noul:", Float64(py=billing.noul))

    var tone = result.choices["tone"]
    print("tone choice:", String(py=tone.choice), "confidence:", Float64(py=tone.confidence))

    var urgency = result.scores["urgency"]
    print("urgency score:", Float64(py=urgency.score), "confidence:", Float64(py=urgency.confidence))

    client.close()
