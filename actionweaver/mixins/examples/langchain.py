from actionweaver import ActionHandlerMixin, action


class LangChainTools(ActionHandlerMixin):
    def verify_lib_installed(self):
        try:
            import langchain
        except ImportError:
            raise ImportError(
                "`langchain` package not found, please run `pip install langchain`"
            )

    @action(name="GoogleSearch")
    def google_search(self, query: str) -> str:
        """
        Perform a Google search using the provided query.

        This action requires `langchain` and `google-api-python-client` installed, and GOOGLE_API_KEY, GOOGLE_CSE_ID environment variables.
        See https://python.langchain.com/docs/integrations/tools/google_search.

        :param query: The search query to be used for the Google search.
        :return: The search results as a string.
        """
        self.verify_lib_installed()

        from langchain.utilities import GoogleSearchAPIWrapper

        search = GoogleSearchAPIWrapper()
        return search.run(query)
