"""Jev CLI. Parses arguments in Mojo and drives the TypeSafe SDK through Python interop.

Usage:
  mojo run src/my_jev/cli.mojo --state <text> --questions <path.json> [options]
  mojo run src/my_jev/cli.mojo --state-file <path> --questions <path.json>

Options:
  --base-url <url>   default http://localhost:8000
  --model <name>     default gemma3:4b-it-qat
  --json             emit the raw answers as JSON instead of a formatted report
"""

from std.sys import argv, exit
from std.python import Python, PythonObject


comptime DEFAULT_BASE_URL = "http://localhost:8000"
comptime DEFAULT_MODEL = "gemma3:4b-it-qat"


@fieldwise_init
struct Options(Copyable, Movable):
    var state: String
    var state_file: String
    var questions_file: String
    var base_url: String
    var model: String
    var as_json: Bool


def parse_args() raises -> Options:
    var options = Options(
        state="",
        state_file="",
        questions_file="",
        base_url=String(DEFAULT_BASE_URL),
        model=String(DEFAULT_MODEL),
        as_json=False,
    )

    var args = argv()
    var i = 1
    while i < len(args):
        var flag = String(args[i])
        if flag == "--json":
            options.as_json = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise Error("missing value for " + flag)
        var value = String(args[i + 1])
        if flag == "--state":
            options.state = value
        elif flag == "--state-file":
            options.state_file = value
        elif flag == "--questions":
            options.questions_file = value
        elif flag == "--base-url":
            options.base_url = value
        elif flag == "--model":
            options.model = value
        else:
            raise Error("unknown flag " + flag)
        i += 2

    if options.questions_file == "":
        raise Error("--questions <path.json> is required")
    if options.state == "" and options.state_file == "":
        raise Error("one of --state or --state-file is required")
    return options^


def load_state(options: Options, json: PythonObject) raises -> PythonObject:
    # A state file may hold either JSON (object/array) or plain text; the API accepts both.
    if options.state_file != "":
        var text = String(open(options.state_file, "r").read())
        try:
            return json.loads(text)
        except:
            return PythonObject(text)
    return PythonObject(options.state)


def report(result: PythonObject) raises:
    var answers = result.answers
    for key in answers:
        var answer = answers[key]
        var kind = String(py=answer.type)
        if kind == "noul":
            print(String(py=key), "(noul) :", Float64(py=answer.noul))
        elif kind == "choice":
            print(
                String(py=key),
                "(choice):",
                String(py=answer.choice),
                "confidence:",
                Float64(py=answer.confidence),
            )
        elif kind == "score":
            print(
                String(py=key),
                "(score) :",
                Float64(py=answer.score),
                "confidence:",
                Float64(py=answer.confidence),
            )
    print("usage    : input=", Int(py=result.usage.input_tokens), " output=", Int(py=result.usage.output_tokens), sep="")


comptime USAGE = """usage: jev-cli --state <text> | --state-file <path>
                   --questions <path.json>
                   [--base-url <url>] [--model <name>] [--json]"""


def main() raises:
    var options: Options
    try:
        options = parse_args()
    except err:
        print("error:", err)
        print(USAGE)
        exit(2)
        raise err

    var typesafe = Python.import_module("typesafe_sdk")
    var json = Python.import_module("json")

    var questions = json.loads(open(options.questions_file, "r").read())
    var state = load_state(options, json)

    # api_key is required by the SDK's config resolution even though the local
    # server does not check it.
    var client = typesafe.TypeSafeClient(
        api_key="local",
        base_url=options.base_url,
        model=options.model,
    )

    var result = client.system_one(state, questions)

    if options.as_json:
        print(String(py=json.dumps(result.raw_http_response.json(), indent=2)))
    else:
        print("model    :", String(py=result.model))
        report(result)

    client.close()
