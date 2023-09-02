from typing import List

from pydantic import BaseModel

from actionweaver import ActionHandlerMixin, action


class Place(BaseModel):
    lat: float
    lng: float
    description: str


class Folium(ActionHandlerMixin):
    def verify_lib_installed(self):
        import importlib

        library_name = "folium"

        try:
            importlib.import_module(library_name)
        except ImportError as e:
            error_msg = (
                f"The '{library_name}' library is not installed. You can install it using:"
                f"pip install {library_name}"
            )
            raise Exception(error_msg) from e

    @action(name="ShowMap")
    def show_map(self, places: List[Place]) -> str:
        """
        Display a map with the provided latitude and longitude coordinates.

        This action requires proper integration with map services and may require API keys or authentication.
        Ensure that relevant documentation and dependencies are properly set up.

        This method requires `folium` installed
        """
        self.verify_lib_installed()

        import folium

        # Calculate the center of all places
        avg_lat = sum(place["lat"] for place in places) / len(places)
        avg_lng = sum(place["lng"] for place in places) / len(places)

        # Create a folium Map centered at the average latitude and longitude
        m = folium.Map(location=[avg_lat, avg_lng])

        # Add markers for each place
        for place in places:
            folium.Marker(
                [place["lat"], place["lng"]], tooltip=place["description"]
            ).add_to(m)

        # Display the map
        display(m)
        return "Map has been displayed"
