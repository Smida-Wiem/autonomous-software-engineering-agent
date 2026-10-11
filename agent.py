"""
Un petit agent d'ingénierie logicielle - version GRATUITE (Groq).

Il a 3 parties :
  PARTIE 1 - les outils : des fonctions Python normales (lire un fichier, écrire un fichier, lancer une commande)
  PARTIE 2 - la description des outils : le texte qui dit au modèle quels outils existent
  PARTIE 3 - la boucle : demander au modèle, lancer l'outil qu'il veut, renvoyer le résultat, recommencer

À lancer dans le dossier du projet :   python agent.py
"""

import json
import os
import subprocess

from openai import OpenAI

# Groq parle le même "langage" que l'API d'OpenAI, donc on utilise la bibliothèque openai
# et on change seulement l'adresse du serveur.
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],  # ta clé, lue dans la variable GROQ_API_KEY
    base_url="https://api.groq.com/openai/v1",
    max_retries=5,  # si la limite gratuite est atteinte, on attend et on réessaie
)
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 20  # sécurité : l'agent s'arrête après 20 tours, même s'il n'a pas fini


# ---------------------------------------------------------------------------
# PARTIE 1 - LES OUTILS. Ce sont des fonctions Python normales. Rien de magique.
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Fichier {path} écrit."


def run_command(command):
    # Sécurité : c'est TOI qui décides si chaque commande a le droit de tourner.
    answer = input(f"   Autoriser cette commande ?  {command}   [y/n] ")
    if answer.strip().lower() != "y":
        return "L'utilisateur a refusé de lancer cette commande."
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=60
    )
    output = result.stdout + result.stderr
    return output[-4000:] or "(la commande n'a rien affiché)"


FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}


# ---------------------------------------------------------------------------
# PARTIE 2 - LA DESCRIPTION DES OUTILS. Le modèle ne voit pas ton code Python.
# Il lit seulement ces descriptions pour savoir ce qu'il a le droit de demander.
# ---------------------------------------------------------------------------

def describe(name, description, parameters):
    """Met une description d'outil dans le format attendu par l'API."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": text}
                    for key, text in parameters.items()
                },
                "required": list(parameters),
            },
        },
    }


TOOLS = [
    describe(
        "read_file",
        "Read a text file and return its full content. "
        "Use it to look at code before you change it.",
        {"path": "Path of the file to read"},
    ),
    describe(
        "write_file",
        "Create a file, or replace a file completely, with the given content. "
        "You must give the whole file content, not only the changed lines.",
        {"path": "Path of the file to write", "content": "The full new content of the file"},
    ),
    describe(
        "run_command",
        "Run a shell command in the current folder and return what it printed "
        "(normal output and errors). Use it to run code, run tests, or list files. "
        "The command stops after 60 seconds.",
        {"command": "The shell command to run"},
    ),
]

SYSTEM_PROMPT = (
    "You are a software engineering agent working in the current folder. "
    "Use your tools to do the task. After you write code, run it to check that it works. "
    "When the task is finished, say what you did in a few simple sentences."
)


# ---------------------------------------------------------------------------
# PARTIE 3 - LA BOUCLE. C'est elle, l'agent.
# ---------------------------------------------------------------------------

def run_agent(task):
    # La liste des messages est la mémoire de l'agent.
    # Elle commence par les consignes, puis ta demande.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- étape {step} ---")

        # 1. On envoie toute la mémoire au modèle et on récupère sa réponse.
        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
        )
        answer = response.choices[0].message
        tool_calls = answer.tool_calls or []

        if answer.content:
            print("AGENT :", answer.content)

        # 2. On range la réponse dans la mémoire.
        memory_line = {"role": "assistant", "content": answer.content or ""}
        if tool_calls:
            memory_line["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in tool_calls
            ]
        messages.append(memory_line)

        # 3. Pas de demande d'outil = le modèle pense que la tâche est finie.
        if not tool_calls:
            print("\nTerminé.")
            return

        # 4. On lance chaque outil demandé et on range le résultat dans la mémoire.
        for call in tool_calls:
            name = call.function.name
            print(f"OUTIL DEMANDÉ : {name}({call.function.arguments})")
            try:
                arguments = json.loads(call.function.arguments)
                output = FUNCTIONS[name](**arguments)
            except Exception as error:
                # Les erreurs retournent aussi au modèle, pour qu'il corrige son erreur.
                output = f"Error: {error}"
            print("RÉSULTAT :", str(output)[:300])
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": str(output)}
            )
        # ... puis on recommence la boucle.

    print("\nArrêt : l'agent a atteint la limite d'étapes.")


if __name__ == "__main__":
    run_agent(input("Que doit faire l'agent ? "))