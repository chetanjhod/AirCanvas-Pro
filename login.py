import streamlit as st


def show_login():

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, #25245a, transparent 40%),
                radial-gradient(circle at 85% 90%, #401b56, transparent 35%),
                #080b16;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 0.3rem;
            max-width: 1000px;
        }

        .login-heading {
            text-align: center;
            font-size: 34px;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 2px;
            line-height: 1.2;
        }

        .login-subheading {
            text-align: center;
            font-size: 14px;
            color: #aeb8d7;
            margin-top: 5px;
            margin-bottom: 16px;
        }

        .login-footer {
            text-align: center;
            color: #9ca8ca;
            font-size: 12px;
            margin-top: 16px;
            padding-bottom: 4px;
        }

        div.stButton > button {
            width: 100%;
            min-height: 42px;
            border-radius: 10px;
            font-weight: 700;
            background: linear-gradient(90deg, #6755f5, #a14df5);
            color: white;
            border: none;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            border: 1px solid #a78bfa;
            color: white;
            box-shadow: 0 4px 16px rgba(103, 85, 245, 0.25);
        }

        div[data-testid="stTextInput"] input {
            border-radius: 9px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .login-heading {
                font-size: 29px;
            }

            .login-subheading {
                margin-bottom: 12px;
            }

            .login-footer {
                margin-top: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.3, 1])

    with center:
        st.markdown(
            '<div class="login-heading">🎨 AirCanvas Pro</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="login-subheading">Your creativity, in the air.</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### Welcome 👋")
            st.write("Enter your username to start creating.")

            username = st.text_input(
                "Username",
                placeholder="Enter your name",
                max_chars=30,
                key="aircanvas_login_username",
            )

            if st.button("🚀 Get Started"):
                if username.strip():
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username.strip()
                    st.rerun()
                else:
                    st.warning("Please enter your username.")

        st.markdown(
            """
            <div class="login-footer">
                <div style="
                    font-size: 21px;
                    font-weight: 800;
                    letter-spacing: 0.5px;
                    color: #8B5CF6;
                ">
                    Created with 💗 by Chetan Jhod
                </div>
                <div style="
                    font-size: 13px;
                    margin-top: 4px;
                    color: #A0A0B8;
                ">
                    AI Drawing Studio
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def require_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "username" not in st.session_state:
        st.session_state["username"] = ""

    if not st.session_state["logged_in"]:
        show_login()
        st.stop()