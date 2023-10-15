from actionweaver.utils import DEFAULT_ACTION_SCOPE

from .orchestration_expr import (
    RequireNext,
    SelectOne,
    _ActionDefault,
    _ActionHandlerLLMInvoke,
    _ActionHandlerRequired,
    _ActionHandlerSelectOne,
)


class ActionOrchestrationParseError(Exception):
    pass


class Orchestration:
    def __init__(self, data: dict = None):
        self.data = {}

        if data is not None:
            self.data |= data

    def pop(self, key, default=None):
        """
        Remove the item with the specified key from the data dictionary and return its value.
        If the key is not found and a default value is provided, return the default value.
        If the key is not found and no default value is provided, raise a KeyError.
        """
        if key in self.data:
            return self.data.pop(key)
        elif default is not None:
            return default
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())

    def items(self):
        return list(self.data.items())

    def get(self, key, default=None):
        return self.data.get(key, default)

    def clear(self):
        self.data.clear()

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    def __eq__(self, other):
        if isinstance(other, Orchestration):
            return self.data == other.data
        return False

    def to_pyvis_network(self):
        """Experimental for debugging purpose"""
        from pyvis.network import Network

        def _generate_node_config(n):
            if isinstance(n, _ActionHandlerLLMInvoke):
                if n.scope == DEFAULT_ACTION_SCOPE:
                    return {
                        "n_id": str(n),
                        "label": "LLM[DEFAULT]",
                        "color": "#005B00",
                        "physics": False,
                    }
                else:
                    return {
                        "n_id": str(n),
                        "label": f"LLM[{n.scope}]",
                        "color": "#00CC99",
                        "physics": False,
                    }
            elif isinstance(n, str):
                return {
                    "n_id": n,
                    "label": None,
                    "color": "#FF7769",
                    "physics": False,
                }
            else:
                raise ActionException("unsupported node type for visualization")

        g = Network(notebook=True, directed=True)

        for node, expr in self.orch_dict.items():
            g.add_node(**_generate_node_config(node))

            if isinstance(expr, _ActionHandlerSelectOne):
                for dest in expr:
                    g.add_node(**_generate_node_config(dest))
                    g.add_edge(
                        _generate_node_config(node)["n_id"],
                        _generate_node_config(dest)["n_id"],
                        color="#00CC99",
                        label="select",
                        physics=False,
                    ),
            elif isinstance(expr, _ActionHandlerRequired):
                g.add_node(**_generate_node_config(expr.action))
                g.add_edge(
                    _generate_node_config(node)["n_id"],
                    _generate_node_config(expr.action)["n_id"],
                    color="#FF0000",
                    label="requires",
                    physics=False,
                ),
        return g


def parse_orchestration_expr(expr):
    orch_dict = Orchestration()

    def get_first_action(l):
        """
        Get first action name from list expression.
        """
        if isinstance(l, (SelectOne, RequireNext)):
            return get_first_action(l[0])
        elif isinstance(l, str):
            return l
        else:
            raise ActionOrchestrationParseError(
                f"Invalid object in orchestration: {l}."
            )

    def get_last_action(l):
        """
        Get last action name from list expression.
        """
        if isinstance(l, SelectOne):
            raise ActionOrchestrationParseError(
                f"Can't decide last action of SelectOne: {l}."
            )
        elif isinstance(l, RequireNext):
            return get_last_action(l[-1])
        elif isinstance(l, str):
            return l
        else:
            raise ActionOrchestrationParseError(
                f"Invalid object in orchestration: {l}."
            )

    def parse(l):
        """
        Parse list expression into orch_dict.
        """

        if isinstance(l, SelectOne):
            curr = get_last_action(l[0])
            parse(l[0])

            next_actions = []
            for n in list(l)[1:]:
                next_actions.append(get_first_action(n))
                parse(n)

            if curr in orch_dict:
                raise ActionOrchestrationParseError(f"Inconsistency caused by {l}")
            orch_dict[curr] = _ActionHandlerSelectOne(next_actions)
        elif isinstance(l, RequireNext):
            parse(l[0])
            prev, curr = get_last_action(l[0]), None
            for n in range(1, len(l)):
                curr = get_first_action(l[n])

                if prev in orch_dict:
                    raise ActionOrchestrationParseError(f"Inconsistency caused by {l}")
                orch_dict[prev] = _ActionHandlerRequired(get_first_action(curr))
                parse(l[n])

                prev = curr
        elif isinstance(l, str):
            return
        else:
            raise ActionOrchestrationParseError(f"Invalid orchestration expr: {l}.")

    parse(expr)
    return orch_dict


# TODO: solve circular import from importing ActionHandlers
def build_orchestration_dict(action_handlers) -> Orchestration:
    """
    Parse orchestration expressions from all actions.
    """
    # orch_dict = {
    #     _ActionHandlerLLMInvoke(DEFAULT_ACTION_SCOPE): _ActionHandlerSelectOne([])
    # }

    orch_dict = Orchestration()
    orch_dict[_ActionHandlerLLMInvoke(DEFAULT_ACTION_SCOPE)] = _ActionHandlerSelectOne(
        []
    )

    for _, action in action_handlers.name_to_action.items():
        if action.orch_expr:
            if action.orch_expr[0] != action.name:
                raise ActionOrchestrationParseError(
                    f"Invalid expression {action.orch_expr} in Action {action.name}, "
                    "The first element of the orchestration expression must be the action name. "
                    "For example, in 'action1', 'SelectOne' should be used as 'SelectOne([action1, ...])', "
                    "and 'RequireNext' should be used as 'RequireNext([action1, ...])'."
                )

            orch_dict.update(**parse_orchestration_expr(action.orch_expr))

        # Parse scope
        if (
            _ActionHandlerLLMInvoke(action.scope) not in orch_dict
            or orch_dict[_ActionHandlerLLMInvoke(action.scope)] is None
        ):  # if scope doesn't exist or map to None like DEFAULT_ACTION_SCOPE, then create a new one
            orch_dict[_ActionHandlerLLMInvoke(action.scope)] = _ActionHandlerSelectOne(
                []
            )
        elif not isinstance(
            orch_dict[_ActionHandlerLLMInvoke(action.scope)],
            _ActionHandlerSelectOne,
        ):
            raise ActionOrchestrationParseError(
                f"The scope {action.scope} of Action {action.name} causes inconsistency in orchestration."
            )
        orch_dict[_ActionHandlerLLMInvoke(action.scope)].append(action.name)

    return orch_dict
