import React, { Component } from "react";

export class SideNavbar extends Component {
  render() {
    return (
      <div>

        <div>
          <div>Translator</div>
          <div>Education</div>
          <div>Settings</div>
        </div>

        <div>
          <button>Sign in</button>
          <button>Register</button>
        </div>
      </div>
    );
  }
}
export default SideNavbar;
